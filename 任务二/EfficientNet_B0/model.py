# model.py

import torch
import torch.nn as nn
from torchvision import models
import config

# ... (get_alexnet 和 get_resnet50 函数的代码保留，此处省略) ...
def get_alexnet(stage):
    weights = models.AlexNet_Weights.DEFAULT
    model = models.alexnet(weights=weights)
    if stage == 1:
        print("配置模型 (阶段 1 - AlexNet): 冻结所有层，只训练分类器最后一层。")
        for param in model.parameters():
            param.requires_grad = False
    elif stage == 2:
        print("配置模型 (阶段 2 - AlexNet): 解冻分类层和部分卷积层进行微调。")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        for layer_num in [8, 10]:
            for param in model.features[layer_num].parameters():
                param.requires_grad = True
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    return model

def get_resnet50(stage):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    if stage == 1:
        print("配置模型 (阶段 1 - ResNet50): 冻结所有层，只训练分类器最后一层。")
        for param in model.parameters():
            param.requires_grad = False
    elif stage == 2:
        print("配置模型 (阶段 2 - ResNet50): 解冻最后的残差块 (layer4) 和分类层进行微调。")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    return model

# 💡 新增: 加载和配置 EfficientNet-B0 的函数
def get_efficientnet_b0(stage):
    """
    加载 EfficientNet-B0 模型并根据阶段配置参数。
    """
    # 加载预训练的 EfficientNet-B0
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if stage == 1:
        # --- 第一阶段：冻结除最后一层外的所有参数 ---
        print("配置模型 (阶段 1 - EfficientNet-B0): 冻结所有层，只训练分类器最后一层。")
        for param in model.parameters():
            param.requires_grad = False

    elif stage == 2:
        # --- 第二阶段：微调 ---
        # 对于 EfficientNet, 一个常见的策略是解冻最后的几个模块
        print("配置模型 (阶段 2 - EfficientNet-B0): 解冻最后的特征提取块和分类层进行微调。")
        for param in model.parameters():
            param.requires_grad = False

        # 解冻 EfficientNet-B0 的最后一个特征块 (features[8])
        for param in model.features[8].parameters():
            param.requires_grad = True

        # 同时解冻分类器模块 (包含 Dropout 和 Linear)
        for param in model.classifier.parameters():
            param.requires_grad = True

    # 替换 EfficientNet 的最终线性层 (classifier[1])
    # 新的层默认 requires_grad=True
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, config.NUM_CLASSES)

    # 将模型移动到指定设备
    model = model.to(config.DEVICE)

    return model