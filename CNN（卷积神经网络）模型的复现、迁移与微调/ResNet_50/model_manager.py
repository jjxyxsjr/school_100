"""
模型管理模块，负责构建和初始化ResNet-50模型，配置优化器和损失函数。
使用预训练权重，冻结主干网络，仅训练替换后的分类头。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class ModelManager:
    """负责模型的构建和初始化"""
    def __init__(self, num_classes, device, learning_rate=0.001):
        self.device = device
        self.model = self._build_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)

    def _build_model(self, num_classes):
        """构建并配置ResNet-50模型"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model.to(self.device)