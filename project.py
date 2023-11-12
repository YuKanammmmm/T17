import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
lower_yellow = np.array([20, 100, 100])  
upper_yellow = np.array([40, 255, 255])  
lower_blue = np.array([110, 100, 100])  
upper_blue = np.array([130, 255, 255])  

cap = cv2.VideoCapture("E:/anaconda3/test2.mp4")  
#cap = cv2.VideoCapture(0)
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
 
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "yellow", (x, y - 5), font, 0.7, (0, 255, 0), 2)
            
            hull = cv2.convexHull(contours[0])
            cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
 
        for cnt2 in contours2:
            (x2, y2, w2, h2) = cv2.boundingRect(cnt2)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), 2)
            cv2.putText(frame, "blue", (x2, y2 - 5), font, 0.7, (0, 0, 255), 2)
            
            hull2 = cv2.convexHull(contours2[0])
            cv2.polylines(frame, [hull2], True, (0, 255, 0), 2)
        num = num + 1
        cv2.imshow("result", frame)
        cv2.imwrite("imgs/%d.jpg"%num, frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
cv2.waitKey(0)
cv2.destroyAllWindows()
