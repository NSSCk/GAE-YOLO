# GAE-YOLO Integrated System
This is an edge device detection algorithm for tomato maturity detection and pest identification, involving Jetson TX2, binocular camera ZED, and YOLO object detection algorithm.The robustness of the code is continuously improving……

The entire project document consists of four main parts, namely：
* GAE-YOLO
* Jetson
* PyQt-UI
* ZED
  
## GAE-YOLO
The implementation of algorithm functions depends on GAE-YOLO, and you need to configure the YOLO environment according to the YOLO official website(https://docs.ultralytics.com/)

* Install the ultralytics library
* `
pip install ultralytics
`
* Download YOLOv11 source code (https://github.com/ultralytics/ultralytics/)
  
You can also configure pytorch and CUDA environments and directly use the code uploaded from this project
### Environment Configuration
* Python3.6/3.7/3.8
* Pytorch
* numpy
* pandas
* matplotlib
* Ubuntu/Windows
* It is best to use GPU training
* Attention: Make appropriate modifications according to your environmental needs
  
### Dataset download address：
* The tomato detection dataset can be obtained in ( https://www.kaggle.com/datasets/andrewmvd/tomato-detection)
* Of course, you can also continue to supplement other datasets for enrichment, as long as the format is the same (https://www.kaggle.com/search?q=tomato+detection+in%3Adatasets)
* The tomato detection dataset can be obtained in (https://universe.roboflow.com/sylhet-agricultural-university/tomato-leaf-diseases-detect)
* Of course, you can also continue to supplement other datasets for enrichment, as long as the format is the same (https://data.mendeley.com/datasets/zfv4jj7855/1)



## Training methods
* Ensure that the dataset is prepared in advance
* Verify the effectiveness of the environment configuration
 `python test-environment.py`
* To train using a single GPU or mutli GPU:
`
train_GhostConv.py
`
* If you want to specify which GPU devices to use, you can add 'CUDA_VISIBLEDEVICES=0.3' before the instruction (for example, I only need to use the first and fourth GPU devices in the device)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## ZED
Establish 3D tracking using Zed
* Firstly, you need to configure the driver environment of the ZED camera according to the ZED official website(Firstly, you need to configure the driver environment of the ZED camera according to the ZED official website)
### Environment Configuration
* os
* platform
* sys
* re
* requests
* argparse
* pathlib
* subprocess
* sys
* math
* shutil
  
* Track the video：
*  `python Record_video.py`
*  Obtain three-dimensional coordinates
*  `python xyz-distance.py`

## Jetson TX2
JetsonTX2 as a edge computing device
* Convert PC to TX2
* `python PC2TX2.py`
* Export to ONNNX format
* `export-ONNX.py`

## PyQt-GUI
GUI software design
### Environment Configuration
* random
* yaml
* numpy
* threading
* pyzed
* PyQt6
  
* Open the interface visualization program
* `tomatoLoadUI.py`




