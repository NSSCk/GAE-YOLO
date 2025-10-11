import math
import random

import cv2  # OpenCV库
import sys

import yaml

# 该部分为实现摄像头调用以及直接显示番茄框
import numpy as np
from threading import Lock
try:
    from ultralytics import YOLO
except ImportError as e:
    print('pip install ultralytics')
try:
    import pyzed.sl as sl
except ImportError as e:
    print(f"zed: {e}")


from zhipuai import ZhipuAI

from PyQt6 import uic
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import QFont, QTextCursor, QTextBlockFormat, QTextCharFormat, QColor, QImage, QPixmap, QMovie, QMouseEvent
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QHeaderView, QGraphicsScene, QGraphicsPixmapItem, \
    QGraphicsSimpleTextItem, QGraphicsView, QTableWidgetItem, QPushButton, QGraphicsDropShadowEffect, QLabel
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget



class AIWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, api_key, user_message):
        super().__init__()
        self.api_key = api_key
        self.user_message = user_message

    def run(self):

        try:
            client = ZhipuAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": self.user_message}],
                stream=False,
            )
            reply = "[Zhipu Qingyan]" + response.choices[0].message.content
        except Exception as e:
            reply = f"API request failed：{str(e)}"
        self.finished.emit(reply)



class MainWindow(QMainWindow):

    class CameraThread(QThread):
        frame_ready = pyqtSignal(np.ndarray)
        status_update = pyqtSignal(str)
        detection_info = pyqtSignal(dict)

        def __init__(self):
            super().__init__()
            self.lock = Lock()
            self.run_signal = False
            self.exit_signal = False
            self.model = None
            self.zed = None
            self.detections = None

        def run(self):
            # 初始化相机
            self.zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_maximum_distance = 50

            if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
                self.status_update.emit("error")
                return
            else:
                self.status_update.emit("run")


            self.model = YOLO('YOLO/TomatoBest.pt')


            image_left = sl.Mat()
            point_cloud = sl.Mat()
            runtime_params = sl.RuntimeParameters()

            while not self.exit_signal:
                if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

                    self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                    self.zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    frame = image_left.get_data()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    results = self.model.predict(frame, imgsz=640)[0]
                    self.detections = results.boxes.cpu().numpy()

                    detection_info = {
                        "count": len(self.detections) if self.detections is not None else 0,
                        "class_name": [],
                        "confidence": [],
                        "xmin": [],
                        "ymin": [],
                        "xmax": [],
                        "ymax": []
                    }
                    if self.detections is not None:
                        for box in self.detections:

                            detection_info["class_name"].append(self.model.names[int(box.cls[0])])
                            detection_info["confidence"].append(float(box.conf[0]))
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detection_info["xmin"].append(x1)
                            detection_info["ymin"].append(y1)
                            detection_info["xmax"].append(x2)
                            detection_info["ymax"].append(y2)

                    self.detection_info.emit(detection_info)


                    if self.detections is not None:
                        for box in self.detections:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls_id = int(box.cls[0])


                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            label = f"{self.model.names[cls_id]} {conf:.2f}"


                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            err, point = point_cloud.get_value(center_x, center_y)
                            if err == sl.ERROR_CODE.SUCCESS and not np.isnan(point[2]):

                                distance = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
                                coord_text = f"Dist:{distance:.2f}m"
                                coord_text2 = f"X:{point[0]:.2f}m Y:{point[1]:.2f}m Z:{point[2]:.2f}m"


                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{self.model.names[cls_id]} {conf:.2f}",
                                            (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame, coord_text,
                                            (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                cv2.putText(frame, coord_text2,
                                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


                    self.detection_info.emit(detection_info)
                    self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            self.zed.close()

        def stop(self):
            with self.lock:
                self.exit_signal = True
            if self.isRunning():
                self.wait(5000)  # 最多等待5秒
            if self.zed.is_opened():
                self.zed.close()

    def __init__(self) -> object:
        super().__init__()
        self.ui = uic.loadUi("./tomato.ui", self)
        self.yolo_model = YOLO("./YOLO/best.pt")
        self.yolo_tomato_model = YOLO("YOLO/TomatoBest.pt")
        self.class_names = self.load_class_names("./YOLO/data.yaml")

        self.cap = None
        self.video_timer = QTimer()
        self.setup_ui()
        self.connect_buttons()
        self.worker = None
        # 摄像头相关属性
        self.camera_thread = self.CameraThread()
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.status_update.connect(self.ui.label_16.setText)
        self.camera_thread.detection_info.connect(self.update_detection_info)

        self.sidebar_visible = False
        self.sidebar_widget = self.ui.findChild(QWidget, "sidebarWidget")
        self.sidebar_widget.setStyleSheet("""
                    QWidget#sidebarWidget {
                        background-color: #eef7f2;
                        border-left: 1px solid #d0d0d0;
                    }
                """)
        self.init_sidebar_position()
        self.setup_sidebar_animation()
        self.ui.pushButton_7.clicked.connect(self.toggle_sidebar)


        shadow_effect = QGraphicsDropShadowEffect(self.ui.sidebarWidget)
        shadow_effect.setBlurRadius(16)
        shadow_effect.setOffset(8, 0)
        shadow_effect.setColor(QColor(0, 0, 0, 64))

        self.ui.sidebarWidget.setGraphicsEffect(shadow_effect)


        if hasattr(self, 'enterwidget'):
            self.enterwidget.hide()
        else:
            print("enterwidget")


        if hasattr(self, 'btn_open_subwindow'):
            self.btn_open_subwindow.clicked.connect(self.show_login_window)
        else:
            print("btn_open_subwindow")

        self.video_paused = False


        self.tomato_class_names = {
            "Mature": {"en": "mature", "zh": "mature"},
            "Immature": {"en": "immature", "zh": "immature"}
        }

        self.ui.tableWidget.itemSelectionChanged.connect(self.on_table_item_selected)


        self.questions = {
            "Late_blight": [
                "How to treat tomato late blight?",
                "Why do tomatoes get late blight?",
                "What is the impact of late blight on tomatoes?"
            ],
            "Early_Blight": [
                "How to treat tomato early blight?",
                "Why do tomatoes get early blight?",
                "What is the impact of early blight on tomatoes?"
            ],
            "Leaf Mold": [
                "How to treat tomato leaf mold?",
                "Why do tomatoes get leaf mold?",
                "What is the impact of leaf mold on tomatoes?"
            ],
            "Target_Spot": [
                "How to treat tomato target spot disease?",
                "Why do tomatoes get target spot disease?",
                "What is the impact of target spot disease on tomatoes?"
            ],
            "black spot": [
                "How to treat tomato black spot disease?",
                "Why do tomatoes get black spot disease?",
                "What is the impact of black spot disease on tomatoes?"
            ],
            "Bacterial Spot": [
                "How to treat tomato bacterial spot disease?",
                "Why do tomatoes get bacterial spot disease?",
                "What is the impact of bacterial spot disease on tomatoes?"
            ]
        }
        self.ui.pushButton_14.hide()
        self.ui.pushButton_15.hide()
        self.ui.pushButton_16.hide()
        self.ui.label_26.hide()
        self.ui.label_27.hide()


        self.selected_tomato_index = -1
        self.detection_results = None
        self.current_image_path = None


        self.video_ended = False
        self.current_frame = None
        self.video_detection_results = None

        self.ui.graphicsView.mousePressEvent = self.on_video_click


        self.ui.pushButton_pause.hide()


        self.ui.pushButton.clicked.connect(self.save_detection_results)
        self.ui.pushButton_pause.clicked.connect(self.toggle_video_playback)

    def on_video_click(self, event):

        if (hasattr(self, 'cap') and self.cap is not None
                and getattr(self, 'video_paused', False)
                and not getattr(self, 'video_ended', False)):
            self.on_video_paused_click(event)

            QGraphicsView.mousePressEvent(self.ui.graphicsView, event)

    def show_login_window(self):

        if hasattr(self, 'enterwidget'):
            self.enterwidget.show()
            self.enterwidget.raise_()

        else:
            print("enterwidget")
    def setup_ui(self):

        self.ui.setFixedSize(918, 619)


        self.set_fonts()


        self.set_background()


        self.ui.tableWidget.setHorizontalHeaderLabels(["order number", "file path", "classification", "confidence", "coordinate"])
        self.ui.tableWidget.setColumnWidth(0, 100)
        self.ui.tableWidget.setColumnWidth(1, 100)
        self.ui.tableWidget.setColumnWidth(2, 100)
        self.ui.tableWidget.setColumnWidth(3, 100)
        self.ui.tableWidget.setColumnWidth(4, 100)
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.ui.tableWidget.setWordWrap(True)
        self.ui.tableWidget.resizeRowsToContents()
        self.ui.tableWidget_2.setHorizontalHeaderLabels(["disease category", "confidence"])
        self.ui.tableWidget_2.setColumnWidth(0, 210)
        self.ui.tableWidget_2.setColumnWidth(1, 85)
        self.ui.tableWidget_2.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.ui.tableWidget_2.resizeRowsToContents()


        self.ui.closeEvent = self.close_event

        self.ui.graphicsView.setMinimumSize(200, 200)

    def set_fonts(self):

        labels1 = [
            self.ui.label_6, self.ui.label_7, self.ui.label_8, self.ui.label_9, self.ui.label_10,
            self.ui.label_12, self.ui.label_13, self.ui.label_14, self.ui.label_15,
            self.label_26, self.label_27
        ]
        for label in labels1:
            label.setFont(QFont("Microsoft YaHei", 11))

        labels2 = [
            self.ui.label_28,self.ui.textEdit, self.ui.lineEdit_3, self.ui.pushButton_14, self.ui.pushButton_15,
            self.ui.pushButton_16
        ]
        for label in labels2:
            label.setFont(QFont("Microsoft YaHei", 13))

        self.ui.label.setFont(QFont("Microsoft YaHei", 18))
        self.ui.label_17.setFont(QFont("Microsoft YaHei", 14))
        self.ui.comboBox.setFont(QFont("Microsoft YaHei", 14))

    def set_background(self):

        transparent_widgets = [
            self.ui.label,
            self.ui.label_6, self.ui.label_7, self.ui.label_8, self.ui.label_9, self.ui.label_10,
            self.ui.label_12, self.ui.label_13, self.ui.label_14, self.ui.label_15,
            self.ui.label_17, self.ui.label_28
        ]
        for widget in transparent_widgets:
            widget.setStyleSheet("background: transparent;")


        white_widgets = [
            self.ui.tableWidget, self.ui.tableWidget_2, self.ui.lineEdit, self.ui.lineEdit_2, self.ui.lineEdit_3,
            self.ui.lineEdit_4, self.ui.pushButton, self.ui.pushButton_2, self.ui.pushButton_3, self.ui.pushButton_4,
            self.ui.pushButton_5,self.ui.pushButton_6, self.ui.pushButton_13, self.ui.graphicsView,self.ui.graphicsView_2,
            self.ui.textEdit, self.ui.comboBox, self.ui.label_16
        ]
        for widget in white_widgets:
            widget.setStyleSheet("background: white;")


        grey_widgets = [
            self.ui.pushButton_14, self.ui.pushButton_15, self.ui.pushButton_16
            ]
        for widget in grey_widgets:
            widget.setStyleSheet("background: #e4dfd7;")

        self.ui.label_26.setStyleSheet("background: #F5F5F5;")
        self.ui.label_27.setStyleSheet("background: #F5F5F5;")


    def connect_buttons(self):

        self.ui.pushButton_3.clicked.connect(self.select_image)
        self.ui.pushButton_4.clicked.connect(self.select_video)
        self.ui.pushButton_5.clicked.connect(self.open_camera)
        self.ui.pushButton_2.clicked.connect(self.close_app)
        self.ui.pushButton_6.clicked.connect(self.send_message)
        self.ui.pushButton_13.clicked.connect(self.select_image_2)
        self.api_key = "4d83067c508741cea168598838b20359.KVZ13emCAkBIPjXx"
        self.ui.pushButton_14.clicked.connect(self.send_question)
        self.ui.pushButton_15.clicked.connect(self.send_question)
        self.ui.pushButton_16.clicked.connect(self.send_question)


    def select_image(self):

        self.release_video_resources()
        self.release_camera_resources()

        self.release_image_resources()

        file_path, _ = QFileDialog.getOpenFileName(
            self.ui,
            "image",
            "",
            "image (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.ui.lineEdit.setText(file_path)
            self.ui.lineEdit_2.setText("")
            self.detect_image_and_display(file_path)
            self.ui.pushButton_pause.hide()

    def detect_image_and_display(self, image_path):

        try:

            start_time = cv2.getTickCount()

            results = self.yolo_tomato_model(image_path, imgsz=640)[0]
            detections = results.boxes.cpu().numpy()

            end_time = cv2.getTickCount()
            detection_time = (end_time - start_time) / cv2.getTickFrequency() * 1000


            self.detection_results = detections
            self.current_image_path = image_path


            frame = cv2.imread(image_path)
            if frame is None:
                raise Exception("error")


            target_count = 0
            if detections is not None:
                target_count = len(detections)
                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])


                    color = (0, 0, 255) if i == self.selected_tomato_index else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


                    label = f"{self.yolo_tomato_model.names[cls_id]} {conf:.2f}"
                    cv2.putText(frame, label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)


            self.ui.label_29.setText(f"{detection_time:.2f}ms")
            self.ui.label_30.setText(f"{target_count}")


            self.ui.tableWidget.setRowCount(target_count)
            for row, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                cls_name_en = self.yolo_tomato_model.names[cls_id]
                cls_name_en_display = self.tomato_class_names.get(cls_name_en, {"en": "unknown"})["en"]


                self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(str(row + 1)))

                self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(image_path))

                self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(f"{cls_name_en_display}"))

                self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(f"{conf:.2f}"))

                coord_text = f"xmin:{x1}, ymin:{y1}, xmax:{x2}, ymax:{y2}"
                self.ui.tableWidget.setItem(row, 4, QTableWidgetItem(coord_text))


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)


            if self.ui.graphicsView.scene() is None:
                scene = QGraphicsScene()
                self.ui.graphicsView.setScene(scene)
            else:
                scene = self.ui.graphicsView.scene()
                scene.clear()

            scene.addPixmap(pixmap)
            self.ui.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)


            self.ui.graphicsView.mousePressEvent = self.on_image_click

        except Exception as e:
            QMessageBox.warning(self.ui, "Detection error", f"Image detection failed: {str(e)}")

            self.display_image_in_view(image_path)

    def display_image_in_view(self, image_path):

        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.ui.graphicsView.setScene(scene)
            self.ui.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        except Exception as e:
            QMessageBox.warning(self.ui, "Display error", f"Unable to display picture: {str(e)}")

    def select_video(self):

        self.release_image_resources()
        self.release_camera_resources()

        self.release_video_resources()

        file_path, _ = QFileDialog.getOpenFileName(
            self.ui,
            "video",
            "",
            "video (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.ui.lineEdit.setText("")
            self.ui.lineEdit_2.setText(file_path)
            self.play_video(file_path)
            self.ui.pushButton_pause.show()

            self.ui.graphicsView.mousePressEvent = self.on_video_click

    def play_video(self, video_path):

        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if self.video_timer.isActive():
            self.video_timer.stop()


        self.current_video_path = video_path
        self.video_paused = False
        self.video_ended = False
        self.total_detection_time = 0
        self.frame_count = 0


        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self.ui, "Error", "Unable to open video file")
            self.release_video_resources()
            return

        # 设置定时器
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_timer.start(int(1000 / (fps if fps > 0 else 30)))

    def update_video_frame(self):

        try:
            ret, frame = self.cap.read()
            if ret:

                self.current_frame = frame.copy()


                start_time = cv2.getTickCount()

                results = self.yolo_tomato_model(frame, imgsz=640)[0]
                detections = results.boxes.cpu().numpy()


                self.video_detection_results = detections


                end_time = cv2.getTickCount()
                detection_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
                self.total_detection_time += detection_time
                self.frame_count += 1


                target_count = len(detections) if detections is not None else 0
                avg_time = self.total_detection_time / self.frame_count if self.frame_count > 0 else 0
                self.ui.label_29.setText(f"{avg_time:.2f}ms")
                self.ui.label_30.setText(f"{target_count}")


                self.ui.tableWidget.setRowCount(target_count)
                for row, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name_en = self.yolo_tomato_model.names[cls_id]
                    cls_name_en_display = self.tomato_class_names.get(cls_name_en, {"en": "unknown"})["en"]


                    self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(str(row + 1)))

                    self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(self.current_video_path))

                    self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(f"{cls_name_en_display}"))

                    self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(f"{conf:.2f}"))

                    coord_text = f"xmin:{x1}, ymin:{y1}, xmax:{x2}, ymax:{y2}"
                    self.ui.tableWidget.setItem(row, 4, QTableWidgetItem(coord_text))


                if detections is not None:
                    for i, box in enumerate(detections):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls_id = int(box.cls[0])


                        color = (0, 0, 255) if i == self.selected_tomato_index else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        label = f"{self.yolo_tomato_model.names[cls_id]} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)


                scene = QGraphicsScene()
                scene.addPixmap(pixmap)


                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                progress = current_frame / total_frames if total_frames > 0 else 0
                progress_text = scene.addText(f"{progress:.1%}", QFont("Arial", 60))
                progress_text.setDefaultTextColor(QColor("#FFFFFF"))
                progress_text.setPos(w - 240, h - 100)



                self.ui.graphicsView.setScene(scene)
                self.ui.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

            else:

                self.video_ended = True
                self.video_timer.stop()
                self.cap.release()
                self.ui.pushButton_pause.hide()
                QMessageBox.information(self.ui, "Play completed", "The video has finished playing")

        except Exception as e:
            self.ui.pushButton_pause.hide()
            print(f" {e}")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            if hasattr(self, 'video_timer') and self.video_timer.isActive():
                self.video_timer.stop()

    def save_detection_results(self):

        if not hasattr(self, 'current_image_path') and not hasattr(self, 'current_video_path'):
            QMessageBox.warning(self.ui, "Save failed", "No detection results can be saved")
            return


        import csv
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui, "Save detection results", f"detection_results_{timestamp}.csv", "CSV file (*.csv)"
        )
        if save_path:
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                headers = [self.ui.tableWidget.horizontalHeaderItem(i).text() for i in range(5)]
                writer.writerow(headers)

                for row in range(self.ui.tableWidget.rowCount()):
                    row_data = [self.ui.tableWidget.item(row, col).text() for col in range(5)]
                    writer.writerow(row_data)
            QMessageBox.information(self.ui, "Successfully saved", f"The detection results have been saved to：{save_path}")

    def toggle_video_playback(self, event):

        if getattr(self, 'video_ended', False):
            return

        if hasattr(self, 'cap') and self.cap is not None:
            self.video_paused = not self.video_paused
            if self.video_paused:
                self.video_timer.stop()
                self.ui.pushButton_pause.setText("Continue")
                self.show_paused_frame()
            else:

                fps = self.cap.get(cv2.CAP_PROP_FPS)
                interval = int(1000 / (fps if fps > 0 else 30))
                self.video_timer.start(interval)
                self.ui.pushButton_pause.setText("Pause")

    def show_paused_frame(self):

        if self.current_frame is None:
            return


        frame = self.current_frame.copy()
        detections = self.video_detection_results


        if detections is not None:
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])


                color = (0, 0, 255) if i == self.selected_tomato_index else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{self.yolo_tomato_model.names[cls_id]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)


        scene = QGraphicsScene()
        scene.addPixmap(pixmap)


        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def on_video_paused_click(self, event):

        if self.video_detection_results is None or self.current_frame is None:
            return


        scene = self.ui.graphicsView.scene()
        if not scene:
            return


        scene_pos = self.ui.graphicsView.mapToScene(event.pos())
        x = scene_pos.x()
        y = scene_pos.y()


        self.ui.tableWidget.clearSelection()


        selected = False
        if self.video_detection_results is not None:
            for i, box in enumerate(self.video_detection_results):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if x1 <= x <= x2 and y1 <= y <= y2:

                    self.selected_tomato_index = i if self.selected_tomato_index != i else -1

                    if self.selected_tomato_index != -1:
                        self.ui.tableWidget.selectRow(i)

                        self.on_table_item_selected()
                    selected = True
                    break


        if not selected:
            self.selected_tomato_index = -1
            self.on_table_item_selected()


        self.show_paused_frame()

    def on_image_click(self, event):

        if self.detection_results is None or self.current_image_path is None:
            return


        scene = self.ui.graphicsView.scene()
        if not scene:
            return

        view_rect = self.ui.graphicsView.viewport().rect()
        scene_rect = self.ui.graphicsView.sceneRect()


        scale_x = scene_rect.width() / view_rect.width()
        scale_y = scene_rect.height() / view_rect.height()


        scene_pos = self.ui.graphicsView.mapToScene(event.pos())
        x = scene_pos.x()
        y = scene_pos.y()


        self.ui.tableWidget.clearSelection()


        selected = False
        if self.detection_results is not None:
            for i, box in enumerate(self.detection_results):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if x1 <= x <= x2 and y1 <= y <= y2:

                    self.selected_tomato_index = i if self.selected_tomato_index != i else -1

                    if self.selected_tomato_index != -1:
                        self.ui.tableWidget.selectRow(i)

                        self.on_table_item_selected()
                    selected = True
                    break


        if not selected:
            self.selected_tomato_index = -1


        self.detect_image_and_display(self.current_image_path)

    def on_table_item_selected(self):

        selected_items = self.ui.tableWidget.selectedItems()
        if not selected_items:

            self.selected_tomato_index = -1
            self.ui.label_31.setText("Not selected")
            self.ui.label_12.setText("xmin:")
            self.ui.label_13.setText("ymin:")
            self.ui.label_14.setText("xmax:")
            self.ui.label_15.setText("ymax:")


            if hasattr(self, 'current_image_path') and self.current_image_path:
                self.detect_image_and_display(self.current_image_path)


            if hasattr(self, 'video_paused') and self.video_paused:
                self.show_paused_frame()
            return


        selected_row = selected_items[0].row()
        self.selected_tomato_index = selected_row


        class_text = self.ui.tableWidget.item(selected_row, 2).text()
        conf_text = self.ui.tableWidget.item(selected_row, 3).text()
        coord_text = self.ui.tableWidget.item(selected_row, 4).text()


        coord_parts = coord_text.split(", ")
        xmin = coord_parts[0].split(":")[1] if len(coord_parts) > 0 else ""
        ymin = coord_parts[1].split(":")[1] if len(coord_parts) > 1 else ""
        xmax = coord_parts[2].split(":")[1] if len(coord_parts) > 2 else ""
        ymax = coord_parts[3].split(":")[1] if len(coord_parts) > 3 else ""

        self.ui.label_32.setText(class_text)
        self.ui.label_33.setText(conf_text)

        # 更新右侧标签显示
        self.ui.label_31.setText("Selected")
        self.ui.label_34.setText(xmin)  # xmin
        self.ui.label_35.setText(ymin)  # ymin
        self.ui.label_36.setText(xmax)  # xmax
        self.ui.label_37.setText(ymax)  # ymax


        if hasattr(self, 'current_image_path') and self.current_image_path:
            self.detect_image_and_display(self.current_image_path)


        if hasattr(self, 'video_paused') and self.video_paused:
            self.show_paused_frame()


    def release_image_resources(self):

        self.current_image_path = None
        self.detection_results = None
        self.selected_tomato_index = -1


        self.ui.tableWidget.setRowCount(0)
        if self.ui.graphicsView.scene():
            self.ui.graphicsView.scene().clear()
            self.ui.graphicsView.setScene(None)

    def release_video_resources(self):

        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None


        if hasattr(self, 'video_timer'):
            if self.video_timer.isActive():
                self.video_timer.stop()
            self.video_timer.deleteLater()
            self.video_timer = QTimer()

        # 清空所有视频相关缓存
        self.current_video_path = None
        self.video_paused = False
        self.video_ended = False
        self.current_frame = None
        self.video_detection_results = None
        self.selected_tomato_index = -1


        self.ui.pushButton_pause.hide()
        self.ui.tableWidget.setRowCount(0)


        if self.ui.graphicsView.scene():
            self.ui.graphicsView.scene().clear()
            self.ui.graphicsView.setScene(None)

    def release_camera_resources(self):

        if self.camera_thread.isRunning():
            self.camera_thread.stop()

        self.ui.label_16.setText("Camera not turned on")
        self.selected_tomato_index = -1

        self.ui.tableWidget.setRowCount(0)

        if self.ui.graphicsView.scene():
            self.ui.graphicsView.scene().clear()



    def select_image_2(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self.ui,
            "image",
            "",
            "image (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.ui.lineEdit_4.setText(file_path)
            self.detect_and_display(file_path)

    def detect_and_display(self, image_path):

        results = self.yolo_model(image_path)


        if hasattr(results[0], 'plot'):
            rendered_image = results[0].plot()
        else:
            raise AttributeError(
                "results“plot”")

        # 更新UI
        self.update_graphics_view_2(rendered_image)
        self.update_table_widget_2(results[0])
    def update_graphics_view_2(self, image):

        if self.ui.graphicsView_2.scene() is None:
            scene = QGraphicsScene()
            self.ui.graphicsView_2.setScene(scene)
        else:
            scene = self.ui.graphicsView_2.scene()
            scene.clear()


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        if pixmap.isNull():
            QMessageBox.warning(self.ui, "Error", "Image loading failed!")
            return


        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)


        self.ui.graphicsView_2.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_class_names(self, yaml_path):

        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        class_names = {}
        for idx, name in enumerate(data['names']):
            class_names[idx] = {"en": name, "zh": "unknow"}  #
            if name == "Bacterial Spot":
                class_names[idx]["zh"] = "1"
            elif name == "Early_Blight":
                class_names[idx]["zh"] = "2"
            elif name == "Healthy":
                class_names[idx]["zh"] = "3"
            elif name == "Late_blight":
                class_names[idx]["zh"] = "4"
            elif name == "Leaf Mold":
                class_names[idx]["zh"] = "5"
            elif name == "Target_Spot":
                class_names[idx]["zh"] = "6"
            elif name == "black spot":
                class_names[idx]["zh"] = "7"
        return class_names

    def update_table_widget_2(self, results):

        if results.boxes:

            detections = results.boxes.cpu().numpy()
            self.ui.tableWidget_2.setRowCount(len(detections))


            disease_confidence = {}
            for box in detections:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names.get(class_id, {"en": "Unknown", "zh": "未知"})["en"]

                if class_name in disease_confidence:
                    disease_confidence[class_name] += confidence
                else:
                    disease_confidence[class_name] = confidence

            detected_classes = set()
            for row, box in enumerate(detections):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names.get(class_id, {"en": "Unknown"})["en"]

                detected_classes.add(class_name)


                self.ui.tableWidget_2.setItem(row, 0, QTableWidgetItem(class_name))
                self.ui.tableWidget_2.setItem(row, 1, QTableWidgetItem(f"{confidence:.2f}"))


                self.ui.tableWidget_2.setColumnWidth(0, 210)
                self.ui.tableWidget_2.setColumnWidth(1, 85)
                self.ui.tableWidget_2.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

                self.update_questions(detected_classes, disease_confidence)
        else:
            self.ui.tableWidget_2.setRowCount(0)
            self.update_questions(set(), {})
            self.ui.label_27.setText("Detection results:")



    def open_camera(self):

        self.release_image_resources()
        self.release_video_resources()

        if not self.camera_thread.isRunning():
            self.camera_thread.start()
            self.ui.label_16.setText("The camera is starting up")
            self.ui.pushButton_pause.hide()
        else:
            QMessageBox.warning(self, "Warning", "The camera is already running!")


    def update_camera_frame(self, frame):

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_detection_info(self, info):

        self.ui.label_7.setText(f"Number of targets:{info.get('count', 0)}")


        if info["count"] > 0:
            self.ui.label_31.setText(f"{info['class_name'][0]}")
            self.ui.label_32.setText(f"{info['confidence'][0]:.2f}")
            self.ui.label_34.setText(f"{info['xmin'][0]}")  # xmin→label_34
            self.ui.label_35.setText(f"{info['ymin'][0]}")  # ymin→label_35
            self.ui.label_36.setText(f"{info['xmax'][0]}")  # xmax→label_36
            self.ui.label_37.setText(f"{info['ymax'][0]}")  # ymax→label_37
        else:
            self.ui.label_31.setText("")
            self.ui.label_32.setText("")
            self.ui.label_34.setText("")
            self.ui.label_35.setText("")
            self.ui.label_36.setText("")
            self.ui.label_37.setText("")


        self.ui.tableWidget.setRowCount(info["count"])
        for row in range(info["count"]):
            cls_name_en = info["class_name"][row]
            cls_name_zh = self.tomato_class_names.get(cls_name_en, {"en": "Unknown"})["en"]
            conf = info["confidence"][row]
            xmin, ymin, xmax, ymax = info["xmin"][row], info["ymin"][row], info["xmax"][row], info["ymax"][row]


            self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(str(row + 1)))

            self.ui.tableWidget.setItem(row, 1, QTableWidgetItem("Camera real-time picture"))

            self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(f"{cls_name_zh}"))

            self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(f"{conf:.2f}"))

            coord_text = f"xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}"
            self.ui.tableWidget.setItem(row, 4, QTableWidgetItem(coord_text))




    def send_message(self):

        current_model = self.ui.comboBox.currentText()
        user_message = self.ui.lineEdit_3.text().strip()

        if not user_message:
            return


        if current_model == "Zhipu Qingyan":
            self._handle_zhipu_message(user_message)
        elif current_model == "Tongyi Qianwen":
            self._handle_local_tyqw_message(user_message)
        else:
            self.append_message("Please select a model", False, "#ff9999")
            self.ui.lineEdit_3.clear()

    def append_message(self, message: str, align_right: bool, background_color: str):

        cursor = self.ui.textEdit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)


        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignmentFlag.AlignRight if align_right else Qt.AlignmentFlag.AlignLeft)


        char_format = QTextCharFormat()
        char_format.setBackground(QColor(background_color))


        cursor.insertBlock(block_format)
        cursor.setCharFormat(char_format)
        cursor.insertText(message + "\n")

    def _handle_zhipu_message(self, user_message):

        self.append_message(user_message, True, "#88b6de")
        self.ui.lineEdit_3.clear()

        self.show_loading_animation()


        self.worker = AIWorker(self.api_key, user_message)
        self.worker.finished.connect(self.handle_ai_reply)
        self.worker.start()

    def show_loading_animation(self):


        self.loading_label = QLabel(self.ui.textEdit)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        movie = QMovie("picture/loading.gif")
        movie.setScaledSize(QSize(144, 124))
        self.loading_label.setMovie(movie)
        movie.start()
        self.loading_label.show()

        text_edit_rect = self.ui.textEdit.contentsRect()
        loading_label_height = self.loading_label.height()
        self.loading_label.move(
            text_edit_rect.left(),
            text_edit_rect.bottom() - loading_label_height
        )

    def remove_loading_animation(self):

        if hasattr(self, 'loading_label') and self.loading_label:
            self.loading_label.deleteLater()
            del self.loading_label

    def handle_ai_reply(self, reply):

        self.remove_loading_animation()

        self.append_message(reply, False, "#FFFFFF")

    def _handle_local_tyqw_message(self, user_message):

        self.append_message(user_message, True, "#88b6de")
        self.ui.lineEdit_3.clear()


        simulated_reply = "[Tongyi Qianwen] Model Response Example (Function to be Implemented)"
        self.append_message(simulated_reply, False, "#FFFFFF")



    def update_questions(self, detected_classes, disease_confidence):

        all_questions = []
        for class_name in detected_classes:
            if class_name in self.questions:
                all_questions.extend(self.questions[class_name])

        if not all_questions:

            self.ui.pushButton_14.hide()
            self.ui.pushButton_15.hide()
            self.ui.pushButton_16.hide()
            self.ui.label_26.hide()
            self.ui.label_27.hide()
            self.ui.label_27.setText("Detection result:")
        else:

            selected_questions = random.sample(all_questions, min(3, len(all_questions)))

            # 设置问题按钮的文本并显示按钮
            if len(selected_questions) > 0:
                self.ui.pushButton_14.setText(selected_questions[0])
                self.ui.pushButton_14.show()
                self.ui.label_26.show()
                self.ui.label_27.show()
            else:
                self.ui.pushButton_14.hide()
                self.ui.label_26.hide()
                self.ui.label_27.hide()

            if len(selected_questions) > 1:
                self.ui.pushButton_15.setText(selected_questions[1])
                self.ui.pushButton_15.show()
                self.ui.label_26.show()
                self.ui.label_27.show()
            else:
                self.ui.pushButton_15.hide()
                self.ui.label_26.hide()
                self.ui.label_27.hide()

            if len(selected_questions) > 2:
                self.ui.pushButton_16.setText(selected_questions[2])
                self.ui.pushButton_16.show()
                self.ui.label_26.show()
                self.ui.label_27.show()
            else:
                self.ui.pushButton_16.hide()
                self.ui.label_26.hide()
                self.ui.label_27.hide()


            if disease_confidence:
                sorted_diseases = sorted(disease_confidence.items(), key=lambda x: x[1], reverse=True)
                detection_result_text = "Detection result:" + "、".join([name for name, _ in sorted_diseases])
            else:
                detection_result_text = "Detection result:"
            self.ui.label_27.setText(detection_result_text)

    def send_question(self):

        sender = self.sender()
        if isinstance(sender, QPushButton):
            question = sender.text()
            if question:
                self.ui.lineEdit_3.setText(question)
                self.send_message()
                self.ui.pushButton_14.hide()
                self.ui.pushButton_15.hide()
                self.ui.pushButton_16.hide()
                self.ui.label_26.hide()
                self.ui.label_27.hide()



    def init_sidebar_position(self):

        sidebar_width = self.sidebar_widget.width()
        win_width = self.ui.width()


        self.sidebar_widget.setGeometry(
            win_width,
            31,
            sidebar_width,
            620
        )
        self.sidebar_widget.raise_()

    def setup_sidebar_animation(self):

        self.sidebar_animation = QPropertyAnimation(
            self.sidebar_widget,
            b"geometry"
        )
        self.sidebar_animation.setDuration(300)
        self.sidebar_animation.setEasingCurve(
            QEasingCurve.Type.OutCubic
        )

    def toggle_sidebar(self):

        self.sidebar_visible = not self.sidebar_visible


        if self.sidebar_animation.state() == QPropertyAnimation.State.Running:
            self.sidebar_animation.stop()


        current_rect = self.sidebar_widget.geometry()
        sidebar_width = current_rect.width()
        win_width = self.ui.width()


        if self.sidebar_visible:

            end_x = win_width - sidebar_width
            self.sidebar_widget.show()
            self.sidebar_widget.raise_()
        else:

            end_x = win_width


        end_rect = QRect(
            end_x,
            31,
            sidebar_width,
            620
        )

        # 配置动画
        self.sidebar_animation.setStartValue(current_rect)
        self.sidebar_animation.setEndValue(end_rect)
        self.sidebar_animation.start()

    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self.sidebar_visible:
            sidebar_width = self.sidebar_widget.width()
            self.sidebar_widget.setGeometry(
                self.ui.width() - sidebar_width,
                0,
                sidebar_width,
                self.ui.height()
            )


    def close_app(self):

        self.ui.close()

    def close_event(self, event):


        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'video_timer') and self.video_timer.isActive():
            self.video_timer.stop()

        if self.camera_thread.isRunning():
            self.camera_thread.stop()

        reply = QMessageBox.question(
            self.ui,
            "Exit",
            "Are you sure you want to exit the program?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    def run(self):

        self.ui.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.run()
    sys.exit(app.exec())