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
* Python3.6/3.7/3.8/3.9
* Pytorch 2.5.1
* numpy 1.26.4
* pandas 2.2.3
* matplotlib 3.9.2
* CUDA Version: 12.1
* CUDNN 9.0.0
* Ubuntu/Windows
* It is best to use GPU training
* Attention: Make appropriate modifications according to your environmental needs
  
### Dataset download address：
* The tomato detection dataset can be obtained in ( https://www.kaggle.com/datasets/andrewmvd/tomato-detection)
* Of course, you can also continue to supplement other datasets for enrichment, as long as the format is the same (https://www.kaggle.com/search?q=tomato+detection+in%3Adatasets)
* The tomato foliar disease detection dataset can be obtained in (https://universe.roboflow.com/sylhet-agricultural-university/tomato-leaf-diseases-detect)
* Of course, you can also continue to supplement other datasets for enrichment, as long as the format is the same (https://data.mendeley.com/datasets/zfv4jj7855/1)



## Training methods
* Ensure that the dataset is prepared in advance
* Verify the effectiveness of the environment configuration
 `python test-environment.py`
* To train using a single GPU or mutli GPU:
* Random seeds:[42, 123, 2023]
`
train_GhostConv.py
`
* If you want to specify which GPU devices to use, you can add 'CUDA_VISIBLEDEVICES=0.3' before the instruction (for example, I only need to use the first and fourth GPU devices in the device)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## ZED
Establish 3D tracking using Zed
* Firstly, you need to configure the driver environment of the ZED camera according to the ZED official website(Firstly, you need to configure the driver environment of the ZED camera according to the ZED official website)
*You can obtain ZED's SDK from here (https://www.stereolabs.com/en-hk/developers/release)
### Environment Configuration
* os
* pyzed==4.0.0
* opencv-python: 4.5.5.64
* numpy: 1.24.3
* torch: 2.0.1
* torchvision: 0.15.2
  
* Track the video：
 `python Record_video.py`
*  Obtain three-dimensional coordinates
 `python xyz-distance.py`

## Jetson TX2
JetsonTX2 as a edge computing device
* Convert PC to TX2
* `python PC2TX2.py`
* Export to ONNNX format
 ` python export-ONNX.py`

## PyQt-GUI
GUI software design
### Environment Configuration

* yaml 0.2.5
* numpy 1.24.4
* threading
* pyzed 4.2 
* PyQt6 6.7.3
 
* Open the interface visualization program
`python tomatoLoadUI.py`

Citation Requirement
Any use, reproduction, modification, or distribution of this code (in whole or in part) — whether for academic, commercial, or personal purposes — must include explicit citation of this project.

License Compliance
This project is licensed under the [Apache License 2.0/GNU GPLv3] (see LICENSE file for details). In addition to the terms outlined in the license, failure to comply with the citation requirement above constitutes a breach of license terms.




