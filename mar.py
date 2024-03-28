import cv2
import numpy as np

# 假设contour是已经找到的蓝色标签的轮廓

# 获取最小外接矩形的四个顶点
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int0(box)

# 1. 将矩形的四个点按距离最近分为两组
group1, group2 = box[:2], box[2:]

# 2. 分别计算每一组点的y值之和，和较小的一组即为位置靠上方的短边
if np.sum(group1[:, 1]) < np.sum(group2[:, 1]):
    top_side, bottom_side = group1, group2
else:
    top_side, bottom_side = group2, group1

# 3. 在靠上方的短边中，始终取x值最小的点作为第一个点A，另一个点作为第二个点B
# 在另一组点中，取x值大的为第三个点C，另一个点为第四点D
A, B = top_side[np.argmin(top_side[:, 0])], top_side[np.argmax(top_side[:, 0])]
C, D = bottom_side[np.argmax(bottom_side[:, 0])], bottom_side[np.argmin(bottom_side[:, 0])]

# 现在我们有了按照ABCD顺序排列的点
sorted_box = np.array([A, B, C, D])

# 4. 对于每个顶点，找到轮廓上离该顶点最近的点
closest_points = []
for p in sorted_box:
    distances = np.sqrt((contour[:, 0, 0] - p[0]) ** 2 + (contour[:, 0, 1] - p[1]) ** 2)
    closest_point_index = np.argmin(distances)
    closest_points.append(contour[closest_point_index][0])

# 将找到的点按顺序组成“points”点集
points = np.array(closest_points)

# 输出点集
print("Sorted points on the contour closest to the rectangle's vertices:", points)
