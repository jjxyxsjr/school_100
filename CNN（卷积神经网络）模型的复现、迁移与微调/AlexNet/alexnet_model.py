# alexnet_project/model/alexnet_model.py

from torchvision import models
import torch.nn as nn


def create_alexnet_model(num_classes=102, pretrained=True):
    """创建一个预训练的、修改了最后一层的AlexNet模型"""

    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)

    # 冻结所有预训练的参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类器的最后一层
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    print("AlexNet 模型创建完成，并已替换最后一层。")
    return model