import cv2
import numpy as np

def check_ripeness(yolo_bbox, image):
    x, y, w, h = yolo_bbox  
    roi = image[y:y+h, x:x+w]  
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_ratio = np.sum(red_mask > 0) / (w * h)

    if red_ratio > 0.7:   return "mature"
    elif red_ratio < 0.3: return "immature"
    else:                 return "overripe"