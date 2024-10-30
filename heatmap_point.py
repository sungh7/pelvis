import itertools
import os
import random
from fastai.vision.all import *


def get_lapa_images(path, max_n=None):
    f = get_files(path, extensions=set(['.jpg']))
    return random.sample(f, max_n) if max_n and len(f) > max_n else f


def get_y(f, lm_idx=None):
    base = os.path.splitext(os.path.basename(f))[0]
    landmarks = f.parents[1]/'landmarks'/(f'{base}.txt')
    with open(landmarks, 'r') as f:
        points = [[float(x) for x in line.strip().split()]
                  for line in f.readlines()]
    points = tensor(points)

    if lm_idx:
        points = points[lm_idx, :]

    return points


# FastAIs `TensorPoint` class comes with a default `PointScaler` transform
# that scales the points to (-1, 1). This little function does the same
# given a point / tensor of points and original size.


def _scale(p, s): return 2 * (p / s) - 1


def heatmap2argmax(heatmap, scale=False):
    N, C, H, W = heatmap.shape
    index = heatmap.view(N, C, 1, -1).argmax(dim=-1)
    pts = torch.cat([index % W, index//W], dim=2)

    if scale:
        scale = tensor([W, H], device=heatmap.device)
        pts = _scale(pts, scale)

    return pts


class Heatmap(TensorImageBase):
    "Heatmap tensor, we can use the type to modify how we display things"
    pass


class HeatmapPoint(TensorPoint):
    """
    A class that mimics TensorPoint, but wraps it so
    we'll be able to override `show` methods with
    a different type.
    """
    pass


class HeatmapTransform(Transform):
    """
    A batch transform that turns TensorPoint instances into Heatmap instances,
    and Heatmap instances into HeatmapPoint instances.

    Used as the last transform after all other transformations.
    """
    # We want the heat map transformation to happen last, so give it a high order value
    order = 999

    def __init__(self, heatmap_size, sigma=10, **kwargs):
        """
        heatmap_size: Size of the heatmap to be created
        sigma: Standard deviation of the Gaussian kernel
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.size = heatmap_size

    def encodes(self, x: TensorPoint):
        # The shape of x is (batch x n_points x 2)
        num_imgs = x.shape[0]
        num_points = x.shape[1]

        maps = Heatmap(torch.zeros(num_imgs, num_points,
                       *self.size, device=x.device))
        for b, c in itertools.product(range(num_imgs), range(num_points)):
            # Note that our point is already scaled to (-1, 1) by PointScaler
            point = x[b][c]
            generate_gaussian(maps[b][c], point[0], point[1], sigma=self.sigma)

        return maps

    def decodes(self, x: Heatmap):
        """
        Decodes a heat map back into a set of points by finding
        the coordinates of their argmax.

        This returns a HeatmapPoint class rather than a TensorPoint
        class, so we can modify how we display the output.
        """
        # Flatten the points along the final axis,
        # and find the argmax per channel
        xy = heatmap2argmax(x, scale=True)
        return HeatmapPoint(xy, source_heatmap=x)


def nmae_argmax(preds, targs):
    # Note that our function is passed two heat maps, which we'll have to
    # decode to get our points. Adding one and dividing by 2 puts us
    # in the range 0...1 so we don't have to rescale for our percentual change.
    preds = 0.5 * (TensorBase(heatmap2argmax(preds, scale=True)) + 1)
    targs = 0.5 * (TensorBase(heatmap2argmax(targs, scale=True)) + 1)

    return ((preds-targs).abs()).mean()


def gaussian_heatmap_loss(pred, targs):
    targs = TensorBase(targs)
    pred = TensorBase(pred)
    return F.mse_loss(pred, targs)


_gaussians = {}


def generate_gaussian(t, x, y, sigma=10):
    """
    Generates a 2D Gaussian point at location x,y in tensor t.

    x should be in range (-1, 1) to match the output of fastai's PointScaler.

    sigma is the standard deviation of the generated 2D Gaussian.
    """
    h, w = t.shape

    # Heatmap pixel per output pixel
    mu_x = int(0.5 * (x + 1.) * w)
    mu_y = int(0.5 * (y + 1.) * h)

    tmp_size = sigma * 3

    # Top-left
    x1, y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

    # Bottom right
    x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
    if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
        return t

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians \
        else tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g

    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)

    t[img_y_min:img_y_max, img_x_min:img_x_max] = \
        g[g_y_min:g_y_max, g_x_min:g_x_max]

    return t
