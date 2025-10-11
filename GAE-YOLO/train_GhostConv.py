
from ultralytics import YOLO

if __name__ == '__main__':

    # 
    # GhostConv(Ghost Convolution)
    model = YOLO('./../your_path/GhostConv.yaml')

    # 使用GPU训练模型
    results = model.train(data='./datasets/data/tomato.yaml',    # modified your path
                          epochs=300, # epochs
                          batch=0.8,  # batch
                          imgsz=640,    # imgsz
                          device=0, # device
                          resume=True, # resume=True
                          project='./runs/tomato_GS',  # 
                          # name='test',  # 
                          )