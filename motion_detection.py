import cv2
cap = cv2.VideoCapture(0)
prev_frame = None
while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray
        continue
    diff = cv2.absdiff(prev_frame,gray)
    prev_frame = gray
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY) #any pixel above 25 becomes white n rest becomes black
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finds the outlines/blobs of the white areas
    for contour in contours:
        if cv2.contourArea(contour) < 500: #ignore tiny movements (like noise)
            continue
        x, y, w, h = cv2.boundingRect(contour) #gets rectangle aroung white blob
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#draws it on frame
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
