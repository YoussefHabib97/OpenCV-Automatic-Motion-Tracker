import cv2

from tracker import *

#Create tracker object

tracker = EuclideanDistTracker()

e1 = cv2.getTickCount()

cap = cv2.VideoCapture("Motorcycle.mp4")

#Object detection from a stable camera (Extracts moving objects from a stable camera)

object_detector = cv2.createBackgroundSubtractorMOG2(history= 100, varThreshold= 50,detectShadows=True)

while True:
    ret, frame = cap.read()

    #height, width, _  = frame.shape
    #print(height,width)

    #Selected Area (Car Video)
    #roi = frame[200 : 720, 300: 800]
    #Selected Area (Motorcycle Video)
    roi = frame[340:720, 500:800]
    vd = frame[340:720, 500:800]
    #cv2.imshow("No effect Video", vd)
    #Whole video
    #roi = frame[1: 720, 1: 1280]


    #Object Detection through mask
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 225, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    detections = []
    for cnt in contours:
        #Calculating area to remove neglegible elements

        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(roi, (x,y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])
            print(detections)

    #Object Tracking
            #cv2.imshow("No effect Video", vd)
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)



    cv2.imshow("ROI", roi)
    #cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
e2 = cv2.getTickCount()

time = (e2-e1) / cv2.getTickFrequency()
print(time)
cap.release()
cv2.destroyAllWindows()