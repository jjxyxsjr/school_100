"""
ModelBuilder class for constructing the EfficientNet-B0 model for flower classification.
Configures the pre-trained model, replaces the classifier, and sets up loss and optimizer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class ModelBuilder:
    def __init__(self, num_classes, learning_rate, device):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device

    def build_efficientnet(self):
        """Builds and configures EfficientNet-B0 with a custom classifier."""
        print("\n正在构建 EfficientNet-B0 模型...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 冻结特征提取层
        for param in model.parameters():
            param.requires_grad = False

        # 替换分类器
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        # 移动到设备
        model = model.to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)

        print("模型构建完成!")
        return model, criterion, optimizer