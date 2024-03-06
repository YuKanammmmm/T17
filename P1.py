from asyncio.windows_events import NULL
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

def draw_lines(img, houghLines, color=[0, 0, 255], thickness=1):
    try:
        for line in houghLines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))
                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))
                cv2.line(img,(x1,y1),(x2,y2),color,thickness)
    except:
        logging.info('hough error')
        
# 计算每个顶点与质心的角度
def angle_with_centroid(point, centroid):
    dx = point[0] - centroid[0]
    dy = point[1] - centroid[1]
    return np.arctan2(dy, dx)

# 设定 fox-2 搜索半径
radius = 30
approx_yellow_prev = None
# fox-2
def find_corresponding_points(points_frame1, points_frame2, radius):
    corresponding_points = []
    for point1 in points_frame1:  # 遍历第一帧的每个顶点
        min_distance = np.inf
        corresponding_point = None
        for point2 in points_frame2:  # 在第二帧中寻找对应点
            distance = np.linalg.norm(point1 - point2)
            if distance < min_distance and distance <= radius:
                min_distance = distance
                corresponding_point = point2
        if corresponding_point is not None:
            corresponding_points.append(corresponding_point)
    return np.array(corresponding_points)

cap = cv2.VideoCapture(0)

while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 设定蓝色和黄色的颜色范围
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # 根据颜色范围创建蓝色和黄色的掩模
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 寻找蓝色和黄色物体的轮廓
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在新窗口中绘制直角坐标系
        coordinate_frame = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # 绘制蓝色物体的边界框并计算长度和宽度
        for contour in contours_blue:
            hull_blue = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull_blue)
            cv2.drawContours(frame, [hull_blue], -1, (255, 0, 0), 2)
            cv2.rectangle(coordinate_frame, (50, 450), (50+w, 450-h), (255, 0, 0), 2)
            cv2.putText(coordinate_frame, f'Width: {w}', (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(coordinate_frame, f'Height: {h}', (50+w, 450-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            # 多边形拟合
            # 拟合精度，长度小于该值的线段将被忽略
            # 该值越小则得到的轮廓中可存在的线段长度越小
            ep = 15
            approx_blue = cv2.approxPolyDP(hull_blue, ep, True)
            cv2.polylines(frame, [approx_blue], True, [0, 255, 0], 2)

        # 设定面积阈值
        area_threshold = 400

        # 绘制黄色物体的边界框并计算长度和宽度
        for contour in contours_yellow:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area > area_threshold:  # 只处理面积大于阈值的轮廓
                hull_yellow = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull_yellow)
                cv2.drawContours(frame, [hull_yellow], -1, (0, 255, 255), 2)
                cv2.rectangle(coordinate_frame, (50, 450), (50+w, 450-h), (0, 255, 255), 2)
                cv2.putText(coordinate_frame, f'Width: {w}', (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(coordinate_frame, f'Height: {h}', (50+w, 450-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
             
                # 多边形拟合
                # 拟合精度，长度小于该值的线段将被忽略
                # 该值越小则得到的轮廓中可存在的线段长度越小
                ep = 15
                approx_yellow = cv2.approxPolyDP(hull_yellow, ep, True)
                cv2.polylines(frame, [approx_yellow], True, [0, 255, 0], 2)
            
                # 如果有上一帧的顶点集，则使用 fox-2 方法进行跟踪和排序
                if approx_yellow_prev is not None:
                    # 重塑顶点集为二维数组，以便使用 find_corresponding_points 函数
                    approx_yellow_reshaped = approx_yellow.reshape((-1, 2))
                    approx_yellow_prev_reshaped = approx_yellow_prev.reshape((-1, 2))

                    # 执行跟踪和排序
                    approx_yellow_this = find_corresponding_points(approx_yellow_prev_reshaped, approx_yellow_reshaped, radius)

                    # 检查是否找到对应点，如果没有，使用当前帧的顶点
                    if approx_yellow_this.size == 0:
                        approx_yellow_this = approx_yellow_reshaped
                else:
                    approx_yellow_this = approx_yellow.reshape((-1, 2))

                # 绘制点集中第一个点
                try:
                    # 确保至少有一个顶点
                    if len(approx_yellow_this) > 0:
                        first_point = approx_yellow_this[0]
                        cv2.circle(frame, tuple(first_point), 30, color=(100, 100, 255), thickness=2)
                except Exception as e:
                    logging.info(f'Cannot draw the first point: {e}')
                
                # 更新 approx_yellow_prev 为当前帧处理后的顶点集，以便下一帧使用
                approx_yellow_prev = approx_yellow_this.reshape((-1, 1, 2))

                # # 绘制点集中第一个点
                # try:
                #     first_point = approx_yellow[0][0]
                #     cv2.circle(frame, tuple(first_point), radius, color=(100, 100, 255), thickness=2)
                # except:
                #     logging.info('can not draw the first point')

                # # fox-2
                # approx_yellow_this = approx_yellow
                # try:
                #     sorted_points = find_corresponding_points(approx_yellow_this, approx_yellow_last, radius)
                #     # 选择排序后的点集中的第一个点
                #     first_point = sorted_points[0]
                #     # 在第一个点的位置绘制一个红色圆圈（假设圆圈半径为 5，线条宽度为 2）
                #     cv2.circle(frame, (int(first_point[0]), int(first_point[1])), radius=5, color=(100, 0, 255), thickness=2)
                # except:
                #     logging.info('fox-2 error')

                # # # 计算几何中心（质心）
                # # centroid = np.mean(approx_yellow[:, 0, :], axis=0)
                # # sorted_points = sorted(approx_yellow[:, 0, :], key=lambda point: angle_with_centroid(point, centroid))
            
                # # # 选择排序后的点集中的第一个点
                # # first_point = sorted_points[0]
                # # # 在第一个点的位置绘制一个红色圆圈（假设圆圈半径为 5，线条宽度为 2）
                # # cv2.circle(frame, (int(first_point[0]), int(first_point[1])), radius=5, color=(100, 0, 255), thickness=2)
            
                # approx_yellow_last = approx_yellow

                    
        # edge
        ## img = cv2.Canny(frame,700,800)
        # hough
        ## rho = 1
        ## theta = np.pi/180
        ## threshold = 140
        ## hough_lines = cv2.HoughLines(img, rho, theta, threshold)
        ## draw_lines(frame, hough_lines)
        
        # 显示结果
        cv2.imshow('Frame', frame)
        cv2.imshow('Coordinates', coordinate_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

