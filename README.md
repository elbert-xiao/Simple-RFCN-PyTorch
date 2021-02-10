## Simple-RFCN-PyTorch
A simple and concise implementation of the RFCN is given.

This project can be run with Pytorch 1.7.

## Results
**1. train on voc2007 & <font color=red>no</font> OHEM**
![results of train on Voc2007](https://github.com/elbert-xiao/Simple-RFCN-PyTorch/blob/master/readme/map_voc2007.png)
**2. train on voc07+12 & OHEM**
![results of train on Voc07+12](https://github.com/elbert-xiao/Simple-RFCN-PyTorch/blob/master/readme/map_voc0712.png)


|                           | Train on voc2007 | Train on voc07+12 |
| :-----------------------: | :--------------: | :---------------: |
| From scratch without OHEM |      71.6%       |         /         |
|  From scratch with OHEM   |      72.5%       |       76.8%       |

## Features
* The results are comparable with those described in the paper(RFCN).
* Very low cuda memory usage (about 3GB(training) and 1.7GB(testing) for resnet101).
* It can be run as pure Python code, no more build affair.


## Requirements:
```requirements.txt
matplotlib==3.2.2
tqdm==4.47.0
numpy==1.18.5
visdom==0.1.8.9
fire==0.3.1
torchnet==0.0.4
opencv_contrib_python==4.5.1.48
scikit_image==0.16.2
torchvision==0.8.1
torch==1.7.0
cupy==8.4.0
Pillow==8.1.0
skimage==0.0
```

## Usage
```shell script
cd [RFCN-pytorch root_dir]
```

**Train:**
```shell script
python -m visdom.server
python train.py RFCN_train
```

Access 'http://localhost:8097/' to view loss and mAP (real-time). <br>
![train fps](https://github.com/elbert-xiao/Simple-RFCN-PyTorch/blob/master/readme/train_fps.png "Fps during training (2080ti)")

**Eval:**<br>
```shell script
python train.py RFCN_eval --load_path='checkPoints/rfcn_voc07_0.725_ohem.pth' --test_num=5000
```
![test fps](https://github.com/elbert-xiao/Simple-RFCN-PyTorch/blob/master/readme/test_fps.png "Fps during testing (2080ti)")

**Predict**<br>
Place the pictures to be predicted in `predict/imgs` folder.<br>
Run command in terminal:<br>
```shell script
python predict.py predict --load_path='checkPoints/rfcn_voc07_0.725_ohem.pth'
```




## Weights & CheckPoints
You can download the [weights of ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) and place it in `weights` folder.

You can download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/191T-sP6Ji1O9A_GMPkRNwTsOM76VOlY2?usp=sharing) and place it in `checkPoints` folder.

## Acknowledgement
This project is writen by [elbert-xiao](https://github.com/elbert-xiao), and thanks to the provider chenyuntc for the project [**simple-faster-rcnn-pytorch**](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)!


If you have any question, please feel free to open an issue.

