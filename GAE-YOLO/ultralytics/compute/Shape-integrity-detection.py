def check_shape(yolo_bbox, image):
    x, y, w, h = yolo_bbox
    gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    circularity = 4 * np.pi * area / (perimeter ** 2)  

    return circularity > 0.8  