# -*- coding: utf-8 -*-
"""模型定义：ResNet18 分割"""
import torch
import torch.nn as nn
from torchvision import models


def get_resnet18():
    """加载 ImageNet 预训练的 ResNet18"""
    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


def split_resnet18(resnet):
    """把 ResNet18 切成前后两半
    前半: conv1 + bn1 + relu + maxpool + layer1 + layer2
    后半: layer3 + layer4
    fc 单独保留
    前半输出: 128 x 28 x 28 的特征图
    """
    front = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2,
    )
    back = nn.Sequential(
        resnet.layer3, resnet.layer4,
    )
    return front, back, resnet.fc
