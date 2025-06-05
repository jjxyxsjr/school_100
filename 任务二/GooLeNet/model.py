# -*- coding: utf-8 -*-
import torch.nn as nn
from torchvision import models
from 任务二.GooLeNet import config


def get_alexnet(pretrained=True):
    """
    加载预训练的AlexNet模型，并替换最后一层以适应我们的任务。
    """
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(num_features, config.NUM_CLASSES)

    return model.to(config.DEVICE)