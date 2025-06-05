# model.py

import torch
import torch.nn as nn
from torchvision import models
import config


def get_alexnet(stage):
    # ... (保留 alexnet 的代码，此处省略) ...
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


# 💡 新增: 加载和配置 ResNet50 的函数
def get_resnet50(stage):
    """
    加载 ResNet50 模型并根据阶段配置参数。
    Args:
        stage (int): 1 表示冻结训练，2 表示微调。
    Returns:
        torch.nn.Module: 配置好的 ResNet50 模型。
    """
    # 加载预训练的 ResNet50
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    if stage == 1:
        # --- 第一阶段：冻结除最后一层外的所有参数 ---
        print("配置模型 (阶段 1 - ResNet50): 冻结所有层，只训练分类器最后一层。")
        for param in model.parameters():
            param.requires_grad = False

    elif stage == 2:
        # --- 第二阶段：解冻部分层进行微调 ---
        # 对于 ResNet, 一个常见的策略是解冻最后的残差块 (layer4)
        print("配置模型 (阶段 2 - ResNet50): 解冻最后的残差块 (layer4) 和分类层进行微调。")
        for param in model.parameters():
            param.requires_grad = False

        # 解冻 layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

    # 替换 ResNet 的最终全连接层 (fc)
    # 新的层默认 requires_grad=True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

    # 将模型移动到指定设备
    model = model.to(config.DEVICE)

    return model