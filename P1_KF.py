from asyncio.windows_events import NULL
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import  deque
from scipy.spatial.distance import euclidean
from Function import *

# 声明全局变量，用于记录标签点集
global_approx_blue   = None
global_approx_yellow = None
# 创建状态记录变量，0=正常排序，1=异常排序
if_error_blue   = 0
if_error_yellow = 0
# kalman filter 初始化追踪点的列表
mybuffer = 256
blue_1 = deque(maxlen=mybuffer)
blue_2 = deque(maxlen=mybuffer)
blue_3 = deque(maxlen=mybuffer)
blue_4 = deque(maxlen=mybuffer)
yellow_1 = deque(maxlen=mybuffer)
yellow_2 = deque(maxlen=mybuffer)
yellow_3 = deque(maxlen=mybuffer)
yellow_4 = deque(maxlen=mybuffer)

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
    i_b = 0

    if global_approx_blue is None and len(points_this_frame) == 4:
        global_approx_blue = points_this_frame
    
    if global_approx_blue is not None:
        # 新建缓存用点集，符合numpy int数据格式
        runningPointList = np.array([], dtype=np.int32).reshape(0, 1, 2)
        n = 0  # 记录是第几次循环的变量yellow
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
                        i_b = n
            # 如果本帧遍历结束后仍未找到上一帧的点的对应点
            if num == 0:
                runningPointList = np.concatenate((runningPointList, [point_afterFox2]), axis=0) # 继承上一帧的点的坐标
                if_err_b = 1 # 记录"出现错误"状态
                i_b = n
            # 对上一帧中的这一点结束处理，进入下一点循环
        # 等待缓存点集写入完成，将其设为当前帧的多边形拟合输出结果
        global_approx_blue = runningPointList
        # 返回错误记录
        return if_err_b,i_b

def fox_2_yellow(points_this_frame, radius):
    # 声明使用全局变量
    global global_approx_yellow
    # 创建状态记录变量，0=正常排序，1=异常排序
    if_err_y = 0
    i_y = 0

    if global_approx_yellow is None and len(points_this_frame) == 4:
        global_approx_yellow = points_this_frame
    
    if global_approx_yellow is not None:
        # 新建缓存用点集，符合numpy int数据格式
        runningPointList = np.array([], dtype=np.int32).reshape(0, 1, 2)
        n = 0  # 记录是第几次循环的变量yellow
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
                        i_y = n
            # 如果本帧遍历结束后仍未找到上一帧的点的对应点
            if num == 0:
                runningPointList = np.concatenate((runningPointList, [point_afterFox2]), axis=0) # 继承上一帧的点的坐标
                if_err_y = 1 # 记录"出现错误"状态
                i_y = n
            # 对上一帧中的这一点结束处理，进入下一点循环
        # 等待缓存点集写入完成，将其设为当前帧的多边形拟合输出结果
        global_approx_yellow = runningPointList
        # 返回错误记录
        return if_err_y,i_y
    
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

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:\毕业设计项目\T17/occlusion_cut_slowslow.mp4")
#
# cv2.namedWindow("Trackbars1")
# cv2.namedWindow("Trackbars2")
#
# cv2.createTrackbar("L - H", "Trackbars1", 0, 179, nothing1)
# cv2.createTrackbar("L - S", "Trackbars1", 0, 255, nothing1)
# cv2.createTrackbar("L - V", "Trackbars1", 0, 255, nothing1)
# cv2.createTrackbar("U - H", "Trackbars1", 179, 179, nothing1)
# cv2.createTrackbar("U - S", "Trackbars1", 255, 255, nothing1)
# cv2.createTrackbar("U - V", "Trackbars1", 255, 255, nothing1)
#
# cv2.createTrackbar("L - H", "Trackbars2", 0, 179, nothing2)
# cv2.createTrackbar("L - S", "Trackbars2", 0, 255, nothing2)
# cv2.createTrackbar("L - V", "Trackbars2", 0, 255, nothing2)
# cv2.createTrackbar("U - H", "Trackbars2", 0, 179, nothing2)
# cv2.createTrackbar("U - S", "Trackbars2", 255, 255, nothing2)
# cv2.createTrackbar("U - V", "Trackbars2", 255, 255, nothing2)


while True:
        ret, frame = cap.read()
        if not ret:
            break

        # GaussianBlur滤波器
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # l_h1 = cv2.getTrackbarPos("L - H", "Trackbars1")
        # l_s1 = cv2.getTrackbarPos("L - S", "Trackbars1")
        # l_v1 = cv2.getTrackbarPos("L - V", "Trackbars1")
        # u_h1 = cv2.getTrackbarPos("U - H", "Trackbars1")
        # u_s1 = cv2.getTrackbarPos("U - S", "Trackbars1")
        # u_v1 = cv2.getTrackbarPos("U - V", "Trackbars1")
        #
        # l_h2 = cv2.getTrackbarPos("L - H", "Trackbars2")
        # l_s2 = cv2.getTrackbarPos("L - S", "Trackbars2")
        # l_v2 = cv2.getTrackbarPos("L - V", "Trackbars2")
        # u_h2 = cv2.getTrackbarPos("U - H", "Trackbars2")
        # u_s2 = cv2.getTrackbarPos("U - S", "Trackbars2")
        # u_v2 = cv2.getTrackbarPos("U - V", "Trackbars2")

        # 设定蓝色和黄色的颜色范围
        # lower_blue = np.array([l_h1, l_s1, l_v1])
        # upper_blue = np.array([u_h1, u_s1, u_v1])
        #
        # lower_yellow = np.array([l_h2, l_s2, l_v2])
        # upper_yellow = np.array([u_h2, u_s2, u_v2])
        lower_blue = np.array([47, 76, 13])
        upper_blue = np.array([84, 255, 255])

        lower_yellow = np.array([27, 74, 40])
        upper_yellow = np.array([34, 255, 255])

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
                cv2.polylines(frame, [global_approx_blue], True, [0, 0, 255], 2)


                
                if_error_blue,i_blue = fox_2_blue(approx_blue, radius)
                # print(f"before_b:\n{approx_blue}")
                # print(f"after__b:\n{global_approx_blue}")
                # print(f"if_err_b: {if_error_blue}\n")

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
                cv2.polylines(frame, [global_approx_yellow], True, [0, 0, 255], 2)
                
                if_error_yellow,i_yellow = fox_2_yellow(approx_yellow, radius)
                # print(f"before_y:\n{approx_yellow}")
                # print(f"after__y:\n{global_approx_yellow}")
                # print(f"if_err_y: {if_error_yellow}\n")

        # ******************* Karmen filter conditions ***********************

        # calculate midpoint coordinates
        # Define the set of identified corner positions定义角坐标点列表
        coordinates = np.array(global_approx_yellow)  # yellow
        coordinates2 = np.array(global_approx_blue)  # blue
        # print("coordinates2",coordinates2)

        # calculate the value of x and y
        # 提取每个点的坐标
        point1 = np.array(coordinates2[0][0])
        point2 = np.array(coordinates2[1][0])
        point3 = np.array(coordinates2[2][0])

        # Calculate the distance between the first point and the second point of the blue tape计算第一个点和第二个点之间的距离
        distance_12 = np.linalg.norm(point2 - point1)

        # Calculate the distance between the second point and the first point of the blue tape计算第二个点和第三个点之间的距离
        distance_23 = np.linalg.norm(point3 - point2)

        # Comparing the distance, the smaller edge is y, and the larger edge is x比较距离大小并赋值给 X 和 Y
        if distance_12 > distance_23:
            XT = distance_12
            YT = distance_23
        else:
            XT = distance_23
            YT = distance_12

            # print("X:", XT)
            # print("Y:", YT)

        # Initialize the list of midpoint coordinates初始化结果列表
        midpoints = []  # yellow
        midpoints2 = []  # blue

        # Initialize the list of midpoint to line distances初始化距离列表
        # yellow tapes distance
        distances_y1 = []
        distances_y2 = []
        distances_y3 = []
        distances_y4 = []
        # blue tapes distance
        distances_b1 = []
        distances_b2 = []
        distances_b3 = []
        distances_b4 = []

        distances_11=[]
        distances_22=[]


        try:
            #Record the number of corner coordinates that detected
            n = len(coordinates)# yellow
            n2 = len(coordinates2)# blue

            # Calculate and store the midpoint coordinates;计算并存储中点坐标
            for i in range(n):
                x_mid = (coordinates[i][0][0] + coordinates[(i + 1) % n][0][0]) / 2
                y_mid = (coordinates[i][0][1] + coordinates[(i + 1) % n][0][1]) / 2
                midpoints.append([x_mid, y_mid])# yellow

            for i in range(n2):
                x_mid = (coordinates2[i][0][0] + coordinates2[(i + 1) % n2][0][0]) / 2
                y_mid = (coordinates2[i][0][1] + coordinates2[(i + 1) % n2][0][1]) / 2
                midpoints2.append([x_mid, y_mid])# blue


            # calculate the distance between midpoints and lines计算角点到四边的距离
            # If four or more corner points are detected
            if len(global_approx_yellow) >= 4 and len(global_approx_blue) >= 4:
                # Defines a list of midpoint coordinates 定义中点坐标列表
                points = np.array(midpoints)# yellow
                points2 = np.array(midpoints2)# blue
                # print("midpoints_yellow", points)
                # print("midpoints2_blue", points2)

                #Every two points form a straight line
                #The slope and intercept of the line where the four edges of the yellow tapes lie
                # yellow tapes四条边所在直线的斜率和截距
                yellow_list = array_to_list(global_approx_yellow)
                edges_properties = rectangle_edges_properties(yellow_list)
                # print("global_approx_yellow", global_approx_yellow)
                # print("approx_list_yellow", yellow_list)
                # print("edges_properties_yellow", edges_properties)

                #The slope and intercept of the line where the four edges of the blue tapes lie
                # blue tapes四条边所在直线的斜率和截距
                blue_list = array_to_list(global_approx_blue)
                edges_properties2 = rectangle_edges_properties(blue_list)
                # print("global_approx_blue", global_approx_blue)
                # print("approx_list_blue", blue_list)
                # print("edges_properties2_blue", edges_properties2)

                # 计算并存储每个点到直线的距离
                # 黄色tapes中点与直线的距离
                distances_y1 = distance_midpoints_lines(points[0], edges_properties2)
                distances_y2 = distance_midpoints_lines(points[1], edges_properties2)
                distances_y3 = distance_midpoints_lines(points[2], edges_properties2)
                distances_y4 = distance_midpoints_lines(points[3], edges_properties2)

                # 蓝色tapes中点与直线的距离
                distances_b1 = distance_midpoints_lines(points2[0], edges_properties)
                distances_b2 = distance_midpoints_lines(points2[1], edges_properties)
                distances_b3 = distance_midpoints_lines(points2[2], edges_properties)
                distances_b4 = distance_midpoints_lines(points2[3], edges_properties)

                #distances_11 = distance_midpoints_lines(global_approx_blue[0], edges_properties)
                #distances_22 = distance_midpoints_lines(global_approx_blue[1], edges_properties)

                # # 打印距离列表
                print("Distances to the line:")
                # print(distances_b1)  # yellow tapes的中点到直线blue1的距离
                print(distances_b2)  # yellow tapes的中点到直线blue2的距离
                # print(distances_b3)  # yellow tapes的中点到直线blue1的距离
                # print(distances_b4)
                # print(distances_y1)  # blue tapes的中点到直线yellow1的距离
                # print(distances_y2)  # blue tapes的中点到直线yellow2的距离
                # print(distances_y3)  # blue tapes的中点到直线yellow1的距离
                # print(distances_y4)
                # #print(distances_11)
                # #print(distances_22)

            else:
                print("没点")

            # **************出界情况*************#

            # 给定直线的 y 坐标
            y_line = 1088
            # 计算每个点到直线 y = 1088 的距离
            distances = []
            for coord in coordinates2:
                # 提取坐标中的 y 值
                y_coord = coord[0][1]
                # 计算距离并添加到 distances 列表中
                distance = np.abs(y_coord - y_line)
                distances.append(distance)
            # 打印结果
            # for i, distance in enumerate(distances):
            #     print(f"点 {i + 1} 到直线的距离为：{distance}")

            # *********************************Kalmanfilter************************************
            #recorded Whether occlusion occurred or not, 1 occluded and 0 not
            # 记录是否发生遮挡，1表示发生遮挡，0表示未发生遮挡
            occlusion_b = 0  # blue tape
            occlusion_y = 0  # yellow

            #Draw the first point of global_approx_blue
            first_point = global_approx_blue[0][0]
            cv2.circle(frame, tuple(first_point), 5, [0, 0, 255], 2)

            #**********test*********
            #print("x / 2 - y",(XT / 2) - YT)
            #print("x / 2",XT / 2)

            # for j in range(len(distances_b4)):
            #     if ((distances_b4[j] > (XT / 2) - YT and distances_b4[j] < XT / 2) or \
            #             (distances_b1[j] > XT / 2 - YT and distances_b1[j] < XT / 2) and \
            #             (distances_b4[j + 1] < YT / 2 or distances_b4[j + 1] < distances_b1[j + 1] )) or \
            #             (if_error_blue == 1 and i_blue == 1):
            #         print("遮挡")
            #     else:
            #         print("未遮挡")


            if distances_b1[2] < 30 or (if_error_blue == 1 and i_blue == 1):

                start_time = time.time()
                #print("blue_list_1",blue_list_1)

                x, y = kf(blue_list_1)
                # 获取循环结束的时间
                end_time = time.time()
                # 计算执行时间
                execution_time = end_time - start_time
                print("blue1循环执行时间为: {:.6f} 秒".format(execution_time))

                blue_1.appendleft([[x, y]])
                blue_list_1 = [tuple(array[0]) for array in blue_1]
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 4)
                global_approx_blue[0] = (x, y)
                occlusion_b = 1
                if x > 1920:
                    cv2.circle(frame, (1920, int(y)), 5, (0, 0, 255), 4)
                if y > 1088:
                    cv2.circle(frame, (int(x), 1088), 5, (0, 0, 255), 4)

                print("遮挡")
            if distances[1] < 1:
                x, y = kf(blue_list_1)
                cv2.circle(frame, (int(x), 1088), 5, (0, 0, 255), 4)
                print("出界")
            else:
                blue_1.appendleft(global_approx_blue[0])
                blue_list_1 = [tuple(array[0]) for array in blue_1]
                #print("global_approx_blue[0]",global_approx_blue[0])
                print("blue_list_1", blue_list_1)
                occlusion_b = 0
                print("未遮挡")

            # 蓝色第二个点的历史记录

            if 10 <distances_b2[2] < 60 or (if_error_blue == 1 and i_blue == 2):
                start_time = time.time()
                x, y = kf(blue_list_2)
                # 获取循环结束的时间
                end_time = time.time()
                # 计算执行时间
                execution_time = end_time - start_time
                print("blue2循环执行时间为: {:.6f} 秒".format(execution_time))
                blue_2.appendleft([[x, y]])
                blue_list_2 = [tuple(array[0]) for array in blue_2]
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 4)
                global_approx_blue[1] = (x, y)
                occlusion_b = 1
                if x > 1920:
                    cv2.circle(frame, (1920, int(y)), 5, (0, 0, 255), 4)
                if y > 1088:
                    cv2.circle(frame, (int(x), 1088), 5, (0, 0, 255), 4)
                print("zhedang")
            else:
                blue_2.appendleft(global_approx_blue[1])
                blue_list_2 = [tuple(array[0]) for array in blue_2]
                occlusion_b = 0
                print("weizhedang")



        except:
            print("coordinates没有值", coordinates)
                    
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

