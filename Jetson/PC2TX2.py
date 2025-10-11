from ultralytics import YOLO


model = YOLO('best.pt')  

# TensorRT
# device=0 
# half=True 
model.export(format='engine', device=0, half=True)


# bash yolo export model=yolov11n.pt format=engine device=0 half=True