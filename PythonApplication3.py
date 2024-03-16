from distutils.dist import DistributionMetadata
from tkinter import END
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2


#点排序
def PointsSorting(inputPoints):
    #初始化  initialization
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = inputPoints
    
    d12 = math.dist((x1, y1), (x2, y2))
    d34 = math.dist((x3, y3), (x4, y4))
    d13 = math.dist((x1, y1), (x3, y3))
    d24 = math.dist((x2, y2), (x4, y4))
    d14 = math.dist((x1, y1), (x4, y4))
    d23 = math.dist((x2, y2), (x3, y3))
    list1 = [d13 + d24, d14 + d23, d12 + d34]
    s1 = list1.index(max(list1))
    X = [x2, x3, x4]
    Y = [y2, y3, y4]
    X1 = [x1, X[s1], X[(s1 + 1) % 3], X[(s1 + 2) % 3]]
    Y1 = [y1, Y[s1], Y[(s1 + 1) % 3], Y[(s1 + 2) % 3]]
    dab1 = math.dist((X1[0], Y1[0]), (X1[1], Y1[1]))
    dbc1 = math.dist((X1[1], Y1[1]), (X1[2], Y1[2]))
    dcd1 = math.dist((X1[2], Y1[2]), (X1[3], Y1[3]))
    dda1 = math.dist((X1[3], Y1[3]), (X1[0], Y1[0]))
    list2 = [dab1, dbc1, dcd1, dda1]
    s2 = list2.index(max(list2))
    if X1[(s2 + 1) % 4] - X1[(s2 + 2) % 4] == 0:
        k1 = 10000
    else:
        k1 = (Y1[(s2 + 1) % 4] - Y1[(s2 + 2) % 4]) / (X1[(s2 + 1) % 4] - X1[(s2 + 2) % 4])
    if X1[(s2 + 2) % 4] - X1[(s2 + 3) % 4] == 0:
        k2 = 10000
    else:
        k2 = (Y1[(s2 + 2) % 4] - Y1[(s2 + 3) % 4]) / (X1[(s2 + 2) % 4] - X1[(s2 + 3) % 4])
    b1 = Y1[s2] - k1 * X1[s2]
    b2 = Y1[(s2 + 2) % 4] - k2 * X1[(s2 + 2) % 4]
    Xd1 = (b1 - b2) / (k1 - k2)
    Yd1 = (k1 * b2 - k2 * b1) / (k1 - k2)
    dd1 = math.dist((X1[s2], Y1[s2]), (Xd1, Yd1))
    if dd1 <= list2[(s2 + 1) % 4]:
        outputpoints = [(X1[s2], Y1[s2]), (Xd1, Yd1), (X1[(s2 + 2) % 4], Y1[(s2 + 2) % 4]), (X1[(s2 + 1) % 4], Y1[(s2 + 1) % 4])]
    elif dd1 > list2[(s2 + 1) % 4]:
        if X1[s2] - X1[(s2 + 3) % 4] == 0:
            k3 = 10000
        else:
            k3 = (Y1[s2 % 4] - Y1[(s2 + 3) % 4]) / (X1[s2] - X1[(s2 + 3) % 4])
        b3 = Y1[(s2 + 1) % 4] - k3 * X1[(s2 + 1) % 4]
        Xd2 = (b3 - b2) / (k2 - k3)
        Yd2 = (k2 * b3 - k3 * b2) / (k2 - k3)
        outputpoints = [(X1[(s2 + 1) % 4], Y1[(s2 + 1) % 4]), (Xd2, Yd2), (X1[(s2 + 3) % 4], Y1[(s2 + 3) % 4]), (X1[s2], Y1[s2])]
    
    return outputpoints

#端点计算函数
def EndpointCalculation(inputPoints, unit, lst, wst, ltopst):
    #初始化  initialization
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = inputPoints

    #计算端点坐标
    xt = ((x1 + x2) / 2 - (x3 + x4) / 2) * ltopst/lst + (x1 + x2) / 2
    yt = ((y1 + y2) / 2 - (y3 + y4) / 2) * ltopst/lst + (y1 + y2) / 2

    #计算端点深度
    c1s = (x2 - x1) * (x2 - x1)+ (y2 - y1) * (y2 - y1)
    d1s = (x3 - x4) * (x3 - x4) + (y3 - y4) * (y3 - y4)
    s1 = wst * unit / math.sqrt(c1s)
    s2 = wst * unit / math.sqrt(d1s)
    st = s1 + (s1 - s2) * ltopst / lst
    
    return xt, yt, st

#转化成笛卡尔坐标系数值
def Coordinate(xt, yt, st, unit, xscreen, yscreen, radianxC):
    radian = (np.pi - radianxC) * 0.5
    xu = xt * math.cos(radian)
    yu = (xscreen - xt) * math.cos(radian)
    zu = yt - 0.5 * yscreen
    x = xu * st / unit
    y = yu * st / unit
    z = zu * st / unit
    return x, y, z

#引入数据 incoming data
#摄像头与屏幕参数 Camera and screen parameters
radianxC = 0.5 * np.pi#摄像头在x轴上的视角范围 Camera perspective range
xscreen = 40#屏幕长度
yscreen = 30#屏幕宽度
#标记数据 Mark data
unit = 20#标准距离 Reference distance
lst = 10#标准长度   Reference length
wst = 2#标准宽度 Reference width 
ltopst = 5#标准尖端长度  Reference tip length

inputpoints = [(15, 10), (15, 20), (17, 10), (17, 20)]#输入点集

inputpoints2 = PointsSorting(inputpoints)

xt, yt, st = EndpointCalculation(inputpoints2, unit, lst, wst, ltopst)#调用函数排序与计算

x, y, z = Coordinate(xt, yt, st, unit, xscreen, yscreen, radianxC)

print(inputpoints2)
print(xt, yt, st)
print(x, y, z)

# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 坐标轴范围
ax.set_xlim([0, 0.75 * xscreen])
ax.set_ylim([0, 0.75 * xscreen])
ax.set_zlim([yscreen * -2 / 3, yscreen * 2 / 3])

# 坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 绘制坐标轴
ax.plot([0, 40], [0, 0], [0, 0], color='r')  # x 轴
ax.plot([0, 0], [0, 40], [0, 0], color='g')  # y 轴
ax.plot([0, 0], [0, 0], [-20, 20], color='b')  # z 轴

# 绘制点
ax.scatter([x], [y], [z], color='k', s=100)

# 显示图形
plt.show()
