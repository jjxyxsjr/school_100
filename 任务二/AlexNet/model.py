# model.py

import torch
import torch.nn as nn
from torchvision import models
import config


def get_alexnet(stage):
    """
    加载 AlexNet 模型并根据阶段配置参数。
    Args:
        stage (int): 1 表示冻结训练，2 表示微调。
    Returns:
        torch.nn.Module: 配置好的 AlexNet 模型。
    """
    # 💡 **主要修正点** 💡
    # 使用新的 'weights' API 来加载预训练模型，取代已弃用的 'pretrained=True'
    # 'DEFAULT' 会加载最新、最好的可用权重。
    weights = models.AlexNet_Weights.DEFAULT
    model = models.alexnet(weights=weights)

    if stage == 1:
        # --- 第一阶段：冻结除最后一层外的所有参数 ---
        print("配置模型 (阶段 1): 冻结所有层，只训练分类器最后一层。")
        for param in model.parameters():
            param.requires_grad = False

    elif stage == 2:
        # --- 第二阶段：解冻部分层进行微调 ---
        print("配置模型 (阶段 2): 解冻分类层和部分卷积层进行微调。")
        for param in model.parameters():
            param.requires_grad = False

        # 解冻所有分类层
        for param in model.classifier.parameters():
            param.requires_grad = True

        # 解冻最后两个卷积块 (features[8] 和 features[10])
        # AlexNet的features: Conv2d, ReLU, MaxPool2d, Conv2d, ReLU, MaxPool2d,
        # Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU, MaxPool2d
        # 我们解冻从第8个模块（第三个卷积层）开始的层
        for layer_num in [8, 10]:
            for param in model.features[layer_num].parameters():
                param.requires_grad = True

    # 替换最后一层全连接层
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, config.NUM_CLASSES)

    # 将模型移动到指定设备
    model = model.to(config.DEVICE)

    return model


if __name__ == '__main__':
    # 测试代码
    print("--- 测试模型加载 (阶段 1) ---")
    model_s1 = get_alexnet(stage=1)
    for name, param in model_s1.named_parameters():
        if param.requires_grad:
            print(f"可训练: {name}")

    print("\n--- 测试模型加载 (阶段 2) ---")
    model_s2 = get_alexnet(stage=2)
    trainable_params_s2 = []
    for name, param in model_s2.named_parameters():
        if param.requires_grad:
            trainable_params_s2.append(name)
    print("可训练层 (阶段 2):")
    for name in trainable_params_s2:
        print(f"- {name}")