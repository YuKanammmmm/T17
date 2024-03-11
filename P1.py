from asyncio.windows_events import NULL
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

# def draw_lines(img, houghLines, color=[0, 0, 255], thickness=1):
#     try:
#         for line in houghLines:
#             for rho,theta in line:
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a*rho
#                 y0 = b*rho
#                 x1 = int(x0 + 2000*(-b))
#                 y1 = int(y0 + 2000*(a))
#                 x2 = int(x0 - 2000*(-b))
#                 y2 = int(y0 - 2000*(a))
#                 cv2.line(img,(x1,y1),(x2,y2),color,thickness)
#     except:
#         logging.info('hough error')

# 声明全局变量，用于记录黄色标签点集
global_approx_yellow = None

# 设定 fox-2 搜索半径
radius = 30

def fox_2_yellow(points_this_frame, radius):
    # 声明使用全局变量
    global global_approx_yellow
    # 创建状态记录变量，0=正常排序，1=异常排序
    if_error = 0

    if global_approx_yellow is None and len(points_this_frame) == 4:
        global_approx_yellow = points_this_frame
    
    if global_approx_yellow is not None:
        # 新建缓存用点集，符合numpy int数据格式
        runningPointList = np.array([], dtype=np.int32).reshape(0, 1, 2)
        # 按上一帧的点集顺序开始遍历
        for point_afterFox2 in global_approx_yellow:
            # 新建状态缓存变量，"等待录入"状态
            num = 0
            # 对于本帧点集中的点
            for point_thisFrame in points_this_frame:
                # 计算本帧各点与上帧指定点的距离
                distance = np.linalg.norm(point_afterFox2 - point_thisFrame)
                # 如果两点之间距离足够近
                if distance <= radius:
                    # 如果有且仅有一个点（未录入过）
                    if num == 0:
                        # 向缓存点集中添加新点
                        runningPointList = np.concatenate((runningPointList, [point_thisFrame]), axis=0)
                        # 成功录入一个点
                        num = 1
                    # 如果有多个点（已录入过）
                    else:
                        if_error = 1 # 忽略此次错误对排序的影响，并记录"出现错误"状态
            # 如果本帧遍历结束后仍未找到上一帧的点的对应点
            if num == 0:
                runningPointList = np.concatenate((runningPointList, [point_afterFox2]), axis=0) # 继承上一帧的点的坐标
                if_error = 1 # 记录"出现错误"状态
            # 对上一帧中的这一点结束处理，进入下一点循环
        # 等待缓存点集写入完成，将其设为当前帧的多边形拟合输出结果
        global_approx_yellow = runningPointList
        # 返回错误记录
        return if_error

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
        # 创建状态记录变量，0=正常排序，1=异常排序
        if_error = 0

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
                
                print(f"before: {approx_yellow}")
                # fox 2 返回一个元组，包括排序后的点集和错误状态记录
                # value1, value2 = my_function()
                # value1, _ = my_function() # 只关心第一个返回值
                # _, value2 = my_function() # 只关心第二个返回值
                if_error = fox_2_yellow(approx_yellow, radius)
                print(f"after_: {global_approx_yellow}")
                print(f"if_err: {if_error}")
            
                # # -----------------------------------------------------------------------------------------------------
                # # 如果有上一帧的顶点集，则使用 fox-2 方法进行跟踪和排序
                # if approx_yellow_prev is not None:
                #     # 重塑顶点集为二维数组，以便使用 find_corresponding_points 函数
                #     approx_yellow_reshaped = approx_yellow.reshape((-1, 2))
                #     approx_yellow_prev_reshaped = approx_yellow_prev.reshape((-1, 2))

                #     # 执行跟踪和排序
                #     approx_yellow_this = find_corresponding_points(approx_yellow_prev_reshaped, approx_yellow_reshaped, radius)

                #     # 检查是否找到对应点，如果没有，使用当前帧的顶点
                #     if approx_yellow_this.size == 0:
                #         approx_yellow_this = approx_yellow_reshaped
                # else:
                #     approx_yellow_this = approx_yellow.reshape((-1, 2))

                # # 绘制点集中第一个点
                # try:
                #     # 确保至少有一个顶点
                #     if len(approx_yellow_this) > 0:
                #         first_point = approx_yellow_this[0]
                #         cv2.circle(frame, tuple(first_point), 30, color=(100, 100, 255), thickness=2)
                # except Exception as e:
                #     logging.info(f'Cannot draw the first point: {e}')
                
                # # 更新 approx_yellow_prev 为当前帧处理后的顶点集，以便下一帧使用
                # approx_yellow_prev = approx_yellow_this.reshape((-1, 1, 2))
                    
        # # edge
        # img = cv2.Canny(frame,700,800)
        # # hough
        # rho = 1
        # theta = np.pi/180
        # threshold = 140
        # hough_lines = cv2.HoughLines(img, rho, theta, threshold)
        # draw_lines(frame, hough_lines)
        
        # 显示结果
        cv2.imshow('Frame', frame)
        cv2.imshow('Coordinates', coordinate_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

