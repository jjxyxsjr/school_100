import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

def build_vgg16_model():
    # 加载 VGG16 预训练模型
    vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)

    # 冻结卷积层参数（迁移学习常规做法）
    for param in vgg16.parameters():
        param.requires_grad = False

    # 修改分类器
    vgg16.classifier[3] = nn.Linear(4096, 1024)
    vgg16.classifier[6] = nn.Linear(1024, 2)

    return vgg16
