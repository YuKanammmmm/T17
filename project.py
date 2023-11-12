import logging
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
lower_yellow = np.array([20, 100, 100])  
upper_yellow = np.array([40, 255, 255])  
lower_blue = np.array([110, 100, 100])  
upper_blue = np.array([130, 255, 255])

# 视频的分辨率
pixiv_x = 1920
pixiv_y = 1088

key_points = []
key_points2 = []

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
        mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue) 

        mask_yellow = cv2.medianBlur(mask_yellow, 7)  
        mask_blue = cv2.medianBlur(mask_blue, 7)  
        mask = cv2.bitwise_or(mask_yellow, mask_blue)
        contours, hierarchy = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hull = cv2.convexHull(contours[0])
        hull2 = cv2.convexHull(contours2[0])
        cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
        cv2.polylines(frame, [hull2], True, (0, 255, 0), 2)

        # gray_y = cv2.cvtColor(mask_yellow, cv2.COLOR_BGR2GRAY)
        # edge_y = cv2.Canny(gray_y, 50, 50)
        # line_y = cv2.HoughLines(edge_y, rho, theta, threshold)
        # draw_hough_lines(frame, line_y)

        # for point in contours[0]:
        for point in hull:
            p = point[0]
            x = p[0].astype(np.int)
            y = p[1].astype(np.int)
            key_points.append((x,y))
            cv2.circle(frame, (x, y), 2, [255,0,0], 2)
            
        for point in hull2:
            p = point[0]
            x = p[0].astype(np.int)
            y = p[1].astype(np.int)
            key_points2.append((x,y))
            cv2.circle(frame, (x, y), 2, [255,0,0], 2)
        
        # for key_point in key_points:
        #     conver_point_temp = []
        #     for theta in range(-90, 90):
        #         x, y = key_point
        #         rad = theta / 180 * np.pi
        #         rho = x * np.cos(rad) + y * np.sin(rad)
        #         conver_point_temp.append((int(theta), int(rho)))
        #     conver_points.append(conver_point_temp)

        # for conver_point in conver_points:
        #     for point in conver_point:
        #         theta, rho = point
        #         key = f"{theta}{rho}"
        #         if key not in point_vote:
        #             point_vote[key] = 0
        #         point_vote[key] += 1
        
        # result = sorted(point_vote.items(), key=lambda d: d[1], reverse=True)[0][0]
        # result = tuple(map(int, result.split()))
        # theta, rho = result
        # radd = theta / 180 * np.pi
        # getY = lambda r, t, x: int((rho-x*np.cos(radd)))/np.sin(radd)
        # cv2.line(frame, (0, getY(rho,theta,0)), (pixiv_x, getY(rho,theta, pixiv_y)), 255)

        num = num + 1
        cv2.imshow("result", frame)
        cv2.imwrite("imgs/%d.jpg"%num, frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
