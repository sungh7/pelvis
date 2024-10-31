import cv2, os, glob, warnings
import matplotlib.pyplot as plt
from heatmap_point import *
from fastai.vision import *
from fastai.vision.all import *
import numpy as np
import matplotlib.patches as mpatches
from util import *

random.seed(42)
model_path = 'models/men_445_500.pkl'

learn = load_learner(model_path)
columns = ['LOPAC', 'LIPTE', 'LIPSS', 'LFHCE', 'LB', 'ROPAC', 'RIPTE', 'RIPSS', 'RFHCE', 'RB']

def calculate_angle_between_points(pt1, pt2, pt3):
    """Calculate the angle at pt2 formed by pt1, pt2, and pt3."""
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def calculate_pelvic_angles(points):
    """Calculate SHARPS, CE, and TONNIS angles."""
    l_opac, l_ipte, l_ipss, l_fhce, l_B = points[:5]
    r_opac, r_ipte, r_ipss, r_fhce, r_B = points[5:]
    
    # SHARPS angles
    sharps_l = calculate_angle_between_points(l_opac, l_ipte, (182, l_ipte[1]))
    sharps_r = calculate_angle_between_points(r_opac, r_ipte, (30, r_ipte[1]))

    # CE angles
    ce_l = calculate_angle_between_points(l_fhce, l_ipte, (182, l_ipte[1]))
    ce_r = calculate_angle_between_points(r_fhce, r_ipte, (30, r_ipte[1]))

    # TONNIS angles
    tonnis_l = calculate_angle_between_points(l_B, l_ipss, (182, l_ipss[1]))
    tonnis_r = calculate_angle_between_points(r_B, r_ipss, (30, r_ipss[1]))

    # Store results
    angles = {
        'SHARPS_L': sharps_l, 'SHARPS_R': sharps_r,
        'CE_L': ce_l, 'CE_R': ce_r,
        'TONNIS_L': tonnis_l, 'TONNIS_R': tonnis_r
    }
    return angles

def process_and_plot_image(image_path, output_size=(640, 640)):
    """
    입력 이미지 파일 경로와 출력 크기를 받아, 해당 이미지의 플롯을 생성해 반환하는 함수.

    Parameters:
    - image_path (str): 이미지 파일 경로.
    - output_size (tuple): 출력 이미지 크기 (width, height).

    Returns:
    - fig: 생성된 플롯 객체.
    """
    
    # 이미지를 로드하고 전처리
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im).resize(output_size)
    im = np.array(im)
    
    # 예측 수행
    y1, _, _ = learn.predict(PILImage.create(im))

    # Set up points based on predictions

    points = [tuple(map(int, y1[j])) for j in range(10)]
    angles = calculate_pelvic_angles(points)

    # Extract points for easier plotting
    l_opac, l_ipte, l_ipss, l_fhce, l_B = points[:5]
    r_opac, r_ipte, r_ipss, r_fhce, r_B = points[5:]

    # IPTE 선 계산
    ipte_a, ipte_b = cal_ab(l_ipte[0], l_ipte[1], r_ipte[0], r_ipte[1])
    l_online = point_line(ipte_a, ipte_b, 182)
    r_online = point_line(ipte_a, ipte_b, 30)

    # 색상 설정
    my_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # 플롯 설정
    fig, ax = plt.subplots(1, 1, figsize=(output_size[0] / 100, output_size[1] / 100))  # DPI 100 기준 크기 조정
    ax.set_ylim(0, output_size[1])
    ax.set_xlim(0, output_size[0])
    
    # 이미지 상에 선 및 각도 표시
    im = cv2.line(im, (0, int(ipte_b)), (output_size[0], int(ipte_a * output_size[0] + ipte_b)), color=(0, 0, 0), thickness=1)
    im = cv2.line(im, l_ipte, l_opac, color=(0, 0, 0), thickness=1)

    # 좌측 각도 계산 및 표시
    angle_l = ang([l_opac, l_ipte], [(182, int(l_online)), l_ipte])
    if angle_l > 90:
        angle_l = 180 - angle_l  # 예각으로 조정
    im = cv2.ellipse(im, center=l_ipte, axes=(15, 15), angle=-angle_l, startAngle=0, endAngle=angle_l, 
                     color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=1)
    cv2.putText(im, f"{angle_l:.2f}", org=(l_ipte[0] - 15, l_ipte[1] + 15), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=1)

    # 우측 각도 계산 및 표시
    im = cv2.line(im, r_ipte, r_opac, color=(0, 0, 0), thickness=1)
    angle_r = ang([r_opac, r_ipte], [(30, int(r_online)), r_ipte])
    if angle_r > 90:
        angle_r = 180 - angle_r  # 예각으로 조정
    im = cv2.ellipse(im, center=r_ipte, axes=(15, 15), angle=angle_r, startAngle=180, endAngle=180-angle_r, 
                     color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=1)
    cv2.putText(im, f"{angle_r:.2f}", org=(r_ipte[0] - 15, r_ipte[1] + 15), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=1)

    # 포인트 라벨 및 스캐터 추가
    for k, (x, y) in enumerate(points):
        ax.scatter(x, output_size[1] - y, color=my_colors[k], s=10)  # my_colors 사용하여 각 포인트에 색상 지정
        cv2.putText(im, str(columns[k]), org=(x + 5, y - 5), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(0, 0, 0), thickness=1)

    # 이미지를 뒤집어 플롯에 추가
    ax.imshow(im[::-1, :], 'gray')

    # 레전드 추가
    patches = [mpatches.Patch(color=my_colors[i], label=label) for i, label in enumerate(columns)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return fig, angles

if __name__ == "__main__":
    image_path = "path/to/your/image.png"  # 실제 이미지 파일 경로
    fig = process_and_plot_image(image_path, output_size=(640, 640))
    fig.show()