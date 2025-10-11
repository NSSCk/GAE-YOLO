import cv2
import numpy as np


def calculate_shape_features(roi):
    # 1.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # 2.
    aspect_ratio = float(roi.shape[1]) / roi.shape[0]

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    roughness = (hull_area - area) / area if area > 0 else 0

    # 3.
    left_right_diff = calculate_symmetry(cnt)

    return {
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'roughness': roughness,
        'symmetry': left_right_diff
    }


def calculate_symmetry(contour):
    # 分割轮廓为左右两部分
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
    left_mask = np.zeros_like(contour)
    right_mask = np.zeros_like(contour)
    for point in contour:
        if point[0][0] < cx:
            left_mask.append(point)
        else:
            right_mask.append(point)
    left_area = cv2.contourArea(left_mask)
    right_area = cv2.contourArea(right_mask)
    return 1 - abs(left_area - right_area) / (left_area + right_area)


def judge_ripeness(shape_features):
    # 归一化得分（示例阈值）
    aspect_score = np.clip((1.2 - abs(shape_features['aspect_ratio'] - 1.1)) / 0.2, 0, 1)
    circ_score = np.clip((shape_features['circularity'] - 0.7) / 0.2, 0, 1)
    sym_score = np.clip((shape_features['symmetry'] - 0.8) / 0.2, 0, 1)
    rough_score = 1 - np.clip(shape_features['roughness'] / 0.2, 0, 1)

    # 加权综合评分
    total_score = 0.3 * aspect_score + 0.4 * circ_score + 0.2 * sym_score + 0.1 * rough_score

    if total_score > 0.7:
        return "mature"
    elif 0.4 <= total_score <= 0.7:
        return "immature"
    else:
        return "overripe"