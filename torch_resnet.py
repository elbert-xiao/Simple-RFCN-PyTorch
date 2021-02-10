import torch
import torchvision
import torch.nn as nn

from torchvision.models.resnet import Bottleneck
from config import opt

def ResNet_BaseNet(pretrained_weights_pth=None):
    # resnet101预训练权重： https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    if pretrained_weights_pth is None:
        # 将conv5的第一个BottleBlock的步长设置为1、3x3卷积设置为空洞卷积
        resnet101 = torchvision.models.resnet101(pretrained=True, replace_stride_with_dilation=[False, False, True])
    else:
        # 加载权重
        print("Loading the resnet101 parameters...")
        resnet101 = torchvision.models.resnet101(pretrained=False, replace_stride_with_dilation=[False, False, True])
        resnet101.load_state_dict(torch.load(pretrained_weights_pth))
        print("Loading complete")

    resnet101_featureLayers = list(resnet101.children())
    resnet101_base1 = nn.Sequential(*resnet101_featureLayers[:7])  # 到layer3(含layer3)

    resnet101_conv_new = resnet101_featureLayers[7]   # 使用空洞卷积的layer4
    if opt.head_ver is not None:
        resnet101_conv_new.add_module('dim_sub', nn.Conv2d(2048, 512, kernel_size=(1, 1), bias=True))   # for vgg16
    else:
        resnet101_conv_new.add_module('dim_sub', nn.Conv2d(2048, 1024, kernel_size=(1, 1), bias=True))
    resnet101_conv_new.add_module('relu', nn.ReLU(inplace=True))

    return resnet101_base1, resnet101_conv_new

