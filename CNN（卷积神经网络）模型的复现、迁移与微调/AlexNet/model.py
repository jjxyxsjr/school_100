"""
模型模块：定义用于花卉分类的神经网络模型。
- 加载并配置预训练的 AlexNet 模型。
- 冻结特征提取层，仅修改分类器以适应目标类别数。
迁移与微调模块：实现模型的迁移学习和微调。迁移最后一层，微调分类器

"""
# model.py
import torch
import torch.nn as nn
from torchvision import models


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerClassifier, self).__init__()
        # 加载预训练 AlexNet
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # 冻结特征提取层
        for param in self.model.parameters():
            param.requires_grad = False
        # 解冻分类器层
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        # 替换最后一层分类器
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def to_device(self, device):
        self.model.to(device)