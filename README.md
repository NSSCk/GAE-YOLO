# GAE-YOLO
This is an edge device detection algorithm for tomato maturity detection and pest identification, involving Jetson TX2, binocular camera ZED, and YOLO object detection algorithm.The robustness of the code is continuously improving……

The entire project document consists of four main parts, namely：
* GAE-YOLO
* Jetson
* PyQt-UI
* ZED
* 
# GAE-YOLO
The implementation of algorithm functions depends on GAE-YOLO, and you need to configure the YOLO environment according to the YOLO official website(https://docs.ultralytics.com/)

# Environment Configuration
* Python3.6/3.7/3.8
* Pytorch
* numpy
* pandas
* matplotlib
* Ubuntu/Windows
* It is best to use GPU training
* Attention: Make appropriate modifications according to your environmental needs
  
# Dataset download address：
* The tomato detection dataset can be obtained in ( https://www.kaggle.com/datasets/andrewmvd/tomato-detection)
* Of course, you can also continue to supplement other datasets for enrichment, as long as the format is the same (https://www.kaggle.com/search?q=tomato+detection+in%3Adatasets)
* The tomato detection dataset can be obtained in (https://universe.roboflow.com/sylhet-agricultural-university/tomato-leaf-diseases-detect)
* Of course, you can also continue to supplement other datasets for enrichment, as long as the format is the same (https://data.mendeley.com/datasets/zfv4jj7855/1)



## Training methods
* Ensure that the dataset is prepared in advance
* To train using a single GPU or CPU:
`
detect train data=datasets/wheat/my data.yaml model=ultralytics/cfg/models/v8/yolov8s CBAM.yaml pretrained=False epochs=300 batch=16 lr0=0.01 resume=True #Need to modify according to one's own actual situation 
`


* If you want to specify which GPU devices to use, you can add 'CUDA_VISIBLEDEVICES=0.3' before the instruction (for example, I only need to use the first and fourth GPU devices in the device)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## Precautions
* When using training scripts, be sure to set '-- data path' to the root directory where you store the 'DRIVE' folder**
* When using prediction scripts, set 'weights_path' to your own generated weight path.

## Load a model
`
from ultralytics import YOLO
model = YOLO("path/to/best.pt")  # load a  model
`
## Validate the model
`
metrics = model.val()  # no arguments needed, dataset and settings remembered
`
## Tracking
* Track the worm and save the video locally, please run track.py
*  Attention: Appropriate modifications need to be made according to your local environment
*  
`
python track.py
`
## behavior-analysis
* To analyze the behavior of the worm, please run behavior-analysis.py
* Attention: Appropriate modifications need to be made according to your local environment
* 
`
python behavior-analysis.py
`
## complex-behavior-analysis
* To analyze the  complex behavior of the worm, please run behavior-analysis.py
* Attention: Appropriate modifications need to be made according to your local environment
* 
`
python COmplex-behavior-analysis.py
`
