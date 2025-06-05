# model.py

import torch
import torch.nn as nn
from torchvision import models

import config


def get_model(device):
    """
    加载预训练的 VGG16 模型并修改最后一层以适应我们的任务。
    """
    # 加载在 ImageNet 上预训练的 VGG16 模型
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 冻结第一阶段的特征提取层
    for param in model.features.parameters():
        param.requires_grad = False

    # 获取最后一个全连接层的输入特征数
    num_ftrs = model.classifier[6].in_features

    # 替换最后一个全连接层
    model.classifier[6] = nn.Linear(num_ftrs, config.NUM_CLASSES)

    # 将模型移动到指定设备
    model = model.to(device)
    print("模型已加载并移动到:", device)

    return model


def unfreeze_model_layers(model):
    """
    为第二阶段微调解冻部分卷积层。
    这里我们解冻最后两个卷积块 (block4, block5)。
    VGG16的 'features' 模块索引:
    - block4: 从索引 17 到 23
    - block5: 从索引 24 到 30
    """
    # 解冻分类器（全连接层）
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 解冻最后两个卷积块
    for param in model.features[17:].parameters():
        param.requires_grad = True

    print("模型的部分卷积层和所有分类层已解冻用于微调。")