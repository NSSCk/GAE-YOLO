from ultralytics import YOLO

# TensorRT
model = YOLO('yolov11n.engine')

# 
results = model('path/to/your/image.jpg')