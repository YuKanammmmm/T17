import cv2
import numpy as np

# 假设蓝色和黄色物体在图像中的像素尺寸
KNOWN_PIXEL_WIDTH = 100
KNOWN_PIXEL_HEIGHT = 100

# 假设蓝色和黄色物体的实际尺寸（单位：厘米）
KNOWN_WIDTH_CM = 10
KNOWN_HEIGHT_CM = 10
KNOWN_DEPTH_CM = 5

def calculate_size(pixel_width, pixel_height):
    # 计算蓝色和黄色物体在图像中的尺寸
    width_cm = (pixel_width * KNOWN_WIDTH_CM) / KNOWN_PIXEL_WIDTH
    height_cm = (pixel_height * KNOWN_HEIGHT_CM) / KNOWN_PIXEL_HEIGHT
    return width_cm, height_cm


cap = cv2.VideoCapture(0)

    # 创建窗口
window_name = '3D Visualization'
cv2.viz.setWindowTitle(window_name, window_name)
cv2.viz.startWindowThread()
viz_window = cv2.viz.Viz3d(window_name)

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

        # 更新3D可视化窗口
        viz_window.removeAllWidgets()

        # 创建蓝色物体的3D框并计算大小
        for contour in contours_blue:
            x, y, w, h = cv2.boundingRect(contour)
            width_cm, height_cm = calculate_size(w, h)
            viz_window.showWidget('Blue Object', cv2.viz.WCube((width_cm, height_cm, KNOWN_DEPTH_CM), color=(255, 0, 0), pose=cv2.viz.makePose((x/100, y/100, 0), (0, 0, 0))))
            cv2.viz.putText3D(viz_window, f'Blue: {width_cm:.2f}m x {height_cm:.2f}m x {KNOWN_DEPTH_CM/100:.2f}m', (x/100, y/100, KNOWN_DEPTH_CM/50), 0.5, (255, 0, 0), 1)

        # 创建黄色物体的3D框并计算大小
        for contour in contours_yellow:
            x, y, w, h = cv2.boundingRect(contour)
            width_cm, height_cm = calculate_size(w, h)
            viz_window.showWidget('Yellow Object', cv2.viz.WCube((width_cm, height_cm, KNOWN_DEPTH_CM), color=(0, 255, 255), pose=cv2.viz.makePose((x/100, y/100, 0), (0, 0, 0))))
            cv2.viz.putText3D(viz_window, f'Yellow: {width_cm:.2f}m x {height_cm:.2f}m x {KNOWN_DEPTH_CM/100:.2f}m', (x/100, y/100, KNOWN_DEPTH_CM/50), 0.5, (0, 255, 255), 1)

        # 显示图像
        cv2.imshow('Frame', frame)

        # 显示3D可视化窗口
        viz_window.spinOnce(1, True)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

