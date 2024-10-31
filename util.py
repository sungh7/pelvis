import math

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180 >= 0:
        # As in if statement
        return 360 - ang_deg
    else: 
        
        return ang_deg
    
#두 점을 지나는 무한한 직선
def cal_ab(x1, y1, x2, y2):
    # y = ax + b
    a = (y2 - y1)/(x2 - x1)
    b = (x2*y1 - x1*y2)/(x2 - x1)
    
    return a, b

# 그 직선을 지나는 점 하나
def point_line(a, b, x):# a,b는 위의 a, b/ x는 임의의 x값
    y = a*x + b
    return y