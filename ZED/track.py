
#!/usr/bin/env python3
import sys
import math
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep

lock = Lock()   #
run_signal = False
exit_signal = False


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))
    x_min = int((xywh[0] - 0.5 * xywh[2]) * im_shape[1])
    y_min = int((xywh[1] - 0.5 * xywh[3]) * im_shape[0])
    x_max = int((xywh[0] + 0.5 * xywh[2]) * im_shape[1])
    y_max = int((xywh[1] + 0.5 * xywh[3]) * im_shape[0])

    output[0][0] = x_min
    output[0][1] = y_min
    output[1][0] = x_max
    output[1][1] = y_min
    output[2][0] = x_min
    output[2][1] = y_max
    output[3][0] = x_max
    output[3][1] = y_max
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections
    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            results = model.predict(img, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0]


            detections = results.boxes.cpu().numpy()
            lock.release()
            run_signal = False
        sleep(0.01) # CPU


def main():
    global image_net, exit_signal, run_signal, detections
    # model = YOLO('D:/../../yolov8n.pt')
    model = YOLO('checkpoints/yolov8n.pt')
    # model = YOLO('checkpoints/best.pt')

    # 初始化相机
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 50

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera")
        exit()


    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size})
    capture_thread.start()


    image_left = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)


            lock.acquire()
            frame = image_left.get_data()
            image_net = frame.copy()
            lock.release()
            run_signal = True


            while run_signal:
                sleep(0.001)


            lock.acquire()
            if detections is not None:
                for box in detections:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    label = f"{model.names[cls_id]} {conf:.2f}"


                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    err, point = point_cloud.get_value(center_x, center_y)
                    if not np.isnan(point[2]):

                        distance = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
                        label += f" {distance:.2f}m"


                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            lock.release()


            cv2.imshow("ZED Detection", frame)
            if cv2.waitKey(10) == 27:  # ESC
                exit_signal = True

    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='checkpoints/yolov11n.pt', help='model path')
    parser.add_argument('--weights', type=str, default='checkpoints/best.pt', help='model path')
    parser.add_argument('--img_size', type=int, default=640, help='inference size')
    opt = parser.parse_args()

    with torch.no_grad():
        main()