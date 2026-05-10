import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,50,150)
    mask = np.zeros_like(canny) #creates black image size as canny
    h,w = canny.shape
    cv2.rectangle(mask,(0,h//2),(w,h),255,-1)# drwas white rectangle on black mask
    roi = cv2.bitwise_and(canny,mask)#combines masks with canny
    lines = cv2.HoughLinesP(roi,1,np.pi/180,50,minLineLength = 50, maxLineGap = 150)
    if lines is not None:  # if opencv found any lines at all
        for line in lines:  # for each line in the list
            x1, y1, x2, y2 = line[0]  # unpack the 2 points of that line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # draw it in green
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
