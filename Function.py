import numpy as np
import time

import cv2

def array_to_list(coordinates_array):
# 将三维数组转换为二维数组
    coordinates_array = coordinates_array.reshape(-1, 2)

# 将 NumPy 数组转换为 Python 列表，并将每个坐标点转换为元组
    coordinates_list = [tuple(point) for point in coordinates_array.tolist()]

    return coordinates_list


def calculate_slope_and_intercept(point1, point2):
    # 计算两点之间的斜率和截距
    x1, y1 = point1
    x2, y2 = point2

    # 斜率的计算
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')  # 处理斜率为无穷大的情况

    # 截距的计算
    intercept = y1 - slope * x1

    return slope, intercept


def rectangle_edges_properties(coordinates):
    edges_properties = []

    # 计算矩形的四条边的斜率和截距
    for i in range(len(coordinates)):
        point1 = coordinates[i]
        point2 = coordinates[(i + 1) % len(coordinates)]  # 闭合矩形的最后一条边连接第一条边

        slope, intercept = calculate_slope_and_intercept(point1, point2)
        edges_properties.append((slope, intercept))

    return edges_properties


def distance_midpoints_lines(midpoint,slope_and_intercept):
    x,y = midpoint
    distances= []
    for s,i in slope_and_intercept:
        distance = abs(s * x - y + i) / np.sqrt(s ** 2 + 1)
        distances.append(distance)
    return distances

def kf(point_list):
    X_posterior_list = []
    # x1,y1,dx, dy
    # 状态转移矩阵
    A = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # 状态观测矩阵
    H = np.eye(4)

    # 过程噪声协方差矩阵 p(w)~N(0,Q)
    # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
    Q = np.eye(4) * 0.05

    # 观察噪声协方差矩阵 p(v)~N(0,R)
    # 观测噪声来自于检测框丢失、重叠等
    R = np.eye(4) * 1

    # 控制输入矩阵B
    B = None

    # 状态估计协方差矩阵P初始化
    P = np.eye(4)
    # 判断是否存在遮挡
    x1, y1 = point_list[149]
    initial_state = np.array([[x1, y1, 0, 0]]).T

    # 状态初始化
    X_posterior = np.array(initial_state)
    P_posterior = np.array(P)
    Z = np.array(initial_state)
    # trace_list = []  # 保存目标box的轨迹

    approx3 = point_list[0:150]

    for i in range(150):
        #print(f'这是第 {i + 1} 次循环。')

        dx = approx3[149 - i][0] - X_posterior[0]
        dy = approx3[149 - i][1] - X_posterior[1]
        #print("approx3[9-i]",approx3[9 - i])
        #print("X_posterior", X_posterior)
        Z[0:2] = np.array(list(approx3[149 - i])).reshape(-1, 1)
        # Z[0:2] = np.array(approx2[0][0]).T
        Z[2::] = np.array([dx, dy])
        # print("Z", Z)

        # 用后验结果
        # -------进行先验估计-------------
        X_prior = np.dot(A, X_posterior)
        # box_prior = xywh_to_xyxy(X_prior[0:4])
        # -------计算状态协方差矩阵P--------
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q
        # -------计算卡尔曼增益------------
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))
        # --------后验估计----------------
        X_posterior_1 = Z - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        # box_posterior = xywh_to_xyxy(X_posterior[0:4])
        # ---------更新状态协方差矩阵P--------
        P_posterior_1 = np.eye(4) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)
        X_prior = X_posterior

        X_posterior_list.append(X_posterior[:2])
        X_posterior_list_1 = [(float(x[0]), float(x[1])) for x in X_posterior_list]

        # 如果IOU匹配失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
        # 此时直接迭代，
        # 先验结果
    X_posterior = np.dot(A, X_posterior)
    x, y = X_posterior[0][0], X_posterior[1][0]

    #plot(approx3, X_posterior_list_1, X_posterior)
    print("****************")
    print("approx3",approx3)
    print("X_posterior_list_1", X_posterior_list_1)
    print("x",X_posterior[0])
    print("y", X_posterior[1])
    #print("X_posterior[:2]",X_posterior[:2])
    return x,y


