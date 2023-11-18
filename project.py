import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX
lower_yellow = np.array([20, 100, 100])  
upper_yellow = np.array([40, 255, 255])  
lower_blue = np.array([110, 100, 100])  
upper_blue = np.array([130, 255, 255])

# 视频的分辨率
pixiv_x = 1920
pixiv_y = 1088

# hough
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

cap = cv2.VideoCapture("C:/Users/28228/Desktop/Subject1.mp4")

if (cap.isOpened()):  
    flag = 1
else:
    flag = 0
num = 0
if (flag):
    while (True):
        ret, frame = cap.read()
       
        if ret == False:  
            break
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)  
        mask_blue   = cv2.inRange(hsv_img, lower_blue, upper_blue)
        mask_yellow_out = cv2.medianBlur(mask_yellow, 7)  
        mask_blue_out   = cv2.medianBlur(mask_blue, 7)
        contours, hierarchy   = cv2.findContours(mask_yellow_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(mask_blue_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hull  = cv2.convexHull(contours[0])
        hull2 = cv2.convexHull(contours2[0])
        # cv2.polylines(frame, [hull],  True, (0, 255, 0), 2)
        # cv2.polylines(frame, [hull2], True, (0, 255, 0), 2)

        # 凸包边缘点集
        # for point in hull:
        #     x,y = point[0]
        #     cv2.circle(frame, (x, y), 2, [255,0,0], 2)
        # for point in hull2:
        #     x,y = point[0]
        #     cv2.circle(frame, (x, y), 2, [255,0,0], 2)
        
        ep = 15
        # 拟合精度，长度小于该值的线段将被忽略
        # 该值越小则得到的轮廓中可存在的线段长度越小
        approx = cv2.approxPolyDP(hull, ep, True)
        cv2.polylines(frame, [approx], True, [0, 255, 0], 2)
        approx2 = cv2.approxPolyDP(hull2, ep, True)
        cv2.polylines(frame, [approx2], True, [0, 255, 0], 2)
        
        # edge
        img = cv2.Canny(frame,700,800)
        # hough
        rho = 1
        theta = np.pi/180
        threshold = 140
        hough_lines = cv2.HoughLines(img, rho, theta, threshold)
        # return
        # img_lines = np.zeros_like(img)
        draw_lines(frame, hough_lines)
        # img_lines = weighted_img(img_lines,img)

        num = num + 1
        cv2.imshow("result", frame)
        cv2.imwrite("imgs/%d.jpg"%num, frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
