from asyncio.windows_events import NULL
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

# 声明全局变量，用于记录标签点集
global_approx_blue   = None
global_approx_yellow = None
# 创建状态记录变量，0=正常排序，1=异常排序
if_error_blue   = 0
if_error_yellow = 0

# 设定多边形拟合面积阈值，小于该值将被过滤
area_threshold = 400
# 设定多边形拟合线段精度，该值越小 线段越短 精度越高 边数越多
ep = 15
# 设定 fox-2 搜索半径
radius = 30

def fox_2_blue(points_this_frame, radius):
    # 声明使用全局变量
    global global_approx_blue
    # 创建状态记录变量，0=正常排序，1=异常排序
    if_err_b = 0

    if global_approx_blue is None and len(points_this_frame) == 4:
        global_approx_blue = points_this_frame
    
    if global_approx_blue is not None:
        # 新建缓存用点集，符合numpy int数据格式
        runningPointList = np.array([], dtype=np.int32).reshape(0, 1, 2)
        # 按上一帧的点集顺序开始遍历
        for point_afterFox2 in global_approx_blue:
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
                        if_err_b = 1 # 忽略此次错误对排序的影响，并记录"出现错误"状态
            # 如果本帧遍历结束后仍未找到上一帧的点的对应点
            if num == 0:
                runningPointList = np.concatenate((runningPointList, [point_afterFox2]), axis=0) # 继承上一帧的点的坐标
                if_err_b = 1 # 记录"出现错误"状态
            # 对上一帧中的这一点结束处理，进入下一点循环
        # 等待缓存点集写入完成，将其设为当前帧的多边形拟合输出结果
        global_approx_blue = runningPointList
        # 返回错误记录
        return if_err_b

def fox_2_yellow(points_this_frame, radius):
    # 声明使用全局变量
    global global_approx_yellow
    # 创建状态记录变量，0=正常排序，1=异常排序
    if_err_y = 0

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
                        if_err_y = 1 # 忽略此次错误对排序的影响，并记录"出现错误"状态
            # 如果本帧遍历结束后仍未找到上一帧的点的对应点
            if num == 0:
                runningPointList = np.concatenate((runningPointList, [point_afterFox2]), axis=0) # 继承上一帧的点的坐标
                if_err_y = 1 # 记录"出现错误"状态
            # 对上一帧中的这一点结束处理，进入下一点循环
        # 等待缓存点集写入完成，将其设为当前帧的多边形拟合输出结果
        global_approx_yellow = runningPointList
        # 返回错误记录
        return if_err_y
    
def nothing1(x):
    pass

def nothing2(x):
    pass

# # 霍夫直线备份
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

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars1")
cv2.namedWindow("Trackbars2")

cv2.createTrackbar("L - H", "Trackbars1", 0, 179, nothing1)
cv2.createTrackbar("L - S", "Trackbars1", 0, 255, nothing1)
cv2.createTrackbar("L - V", "Trackbars1", 0, 255, nothing1)
cv2.createTrackbar("U - H", "Trackbars1", 179, 179, nothing1)
cv2.createTrackbar("U - S", "Trackbars1", 255, 255, nothing1)
cv2.createTrackbar("U - V", "Trackbars1", 255, 255, nothing1)

cv2.createTrackbar("L - H", "Trackbars2", 0, 179, nothing2)
cv2.createTrackbar("L - S", "Trackbars2", 0, 255, nothing2)
cv2.createTrackbar("L - V", "Trackbars2", 0, 255, nothing2)
cv2.createTrackbar("U - H", "Trackbars2", 0, 179, nothing2)
cv2.createTrackbar("U - S", "Trackbars2", 255, 255, nothing2)
cv2.createTrackbar("U - V", "Trackbars2", 255, 255, nothing2)

while True:
        ret, frame = cap.read()
        if not ret:
            break

        # GaussianBlur滤波器
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h1 = cv2.getTrackbarPos("L - H", "Trackbars1")
        l_s1 = cv2.getTrackbarPos("L - S", "Trackbars1")
        l_v1 = cv2.getTrackbarPos("L - V", "Trackbars1")
        u_h1 = cv2.getTrackbarPos("U - H", "Trackbars1")
        u_s1 = cv2.getTrackbarPos("U - S", "Trackbars1")
        u_v1 = cv2.getTrackbarPos("U - V", "Trackbars1")

        l_h2 = cv2.getTrackbarPos("L - H", "Trackbars2")
        l_s2 = cv2.getTrackbarPos("L - S", "Trackbars2")
        l_v2 = cv2.getTrackbarPos("L - V", "Trackbars2")
        u_h2 = cv2.getTrackbarPos("U - H", "Trackbars2")
        u_s2 = cv2.getTrackbarPos("U - S", "Trackbars2")
        u_v2 = cv2.getTrackbarPos("U - V", "Trackbars2")

        # 设定蓝色和黄色的颜色范围
        lower_blue = np.array([l_h1, l_s1, l_v1])
        upper_blue = np.array([u_h1, u_s1, u_v1])

        lower_yellow = np.array([l_h2, l_s2, l_v2])
        upper_yellow = np.array([u_h2, u_s2, u_v2])

        # # 设定蓝色和黄色的颜色范围
        # lower_blue = np.array([100, 100, 100])
        # upper_blue = np.array([140, 255, 255])
        # lower_yellow = np.array([20, 100, 100])
        # upper_yellow = np.array([30, 255, 255])

        # 根据颜色范围创建蓝色和黄色的掩模
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 寻找蓝色和黄色物体的轮廓
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # 在新窗口中绘制直角坐标系
        # coordinate_frame = np.zeros((500, 500, 3), dtype=np.uint8)

        # 绘制蓝色物体的边界框并计算长度和宽度
        for contour in contours_blue:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area > area_threshold:  # 只处理面积大于阈值的轮廓
                hull_blue = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull_blue)
                cv2.drawContours(frame, [hull_blue], -1, (255, 0, 0), 2)
                # cv2.rectangle(coordinate_frame, (50, 450), (50+w, 450-h), (255, 0, 0), 2)
                # cv2.putText(coordinate_frame, f'Width: {w}', (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(coordinate_frame, f'Height: {h}', (50+w, 450-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
                # 多边形拟合
                approx_blue = cv2.approxPolyDP(hull_blue, ep, True)
                cv2.polylines(frame, [approx_blue], True, [0, 255, 0], 2)
                
                if_error_blue = fox_2_blue(approx_blue, radius)
                print(f"before_b:\n{approx_blue}")
                print(f"after__b:\n{global_approx_blue}")
                print(f"if_err_b: {if_error_blue}\n")

        # 绘制黄色物体的边界框并计算长度和宽度
        for contour in contours_yellow:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area > area_threshold:  # 只处理面积大于阈值的轮廓
                hull_yellow = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull_yellow)
                cv2.drawContours(frame, [hull_yellow], -1, (0, 255, 255), 2)
                # cv2.rectangle(coordinate_frame, (50, 450), (50+w, 450-h), (0, 255, 255), 2)
                # cv2.putText(coordinate_frame, f'Width: {w}', (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(coordinate_frame, f'Height: {h}', (50+w, 450-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
             
                # 多边形拟合
                approx_yellow = cv2.approxPolyDP(hull_yellow, ep, True)
                cv2.polylines(frame, [approx_yellow], True, [0, 255, 0], 2)
                
                if_error_yellow = fox_2_yellow(approx_yellow, radius)
                print(f"before_y:\n{approx_yellow}")
                print(f"after__y:\n{global_approx_yellow}")
                print(f"if_err_y: {if_error_yellow}\n")
                    
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
        # cv2.imshow('Coordinates', coordinate_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

