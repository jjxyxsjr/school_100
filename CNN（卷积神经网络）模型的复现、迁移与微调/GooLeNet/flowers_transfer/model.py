# flowers_transfer/model.py

import torch
import torch.nn as nn
import torchvision.models as models

# 从 config.py 导入配置
import config


def get_googlenet_model(num_classes=None, pretrained=None, freeze_extractor=None, aux_logits_user_preference=True):
    """
    加载并配置 GoogLeNet 模型。

    参数:
        num_classes (int, optional): 输出类别的数量。默认为 config.NUM_CLASSES。
        pretrained (bool, optional): 是否加载 ImageNet 预训练权重。默认为 config.USE_PRETRAINED。
        freeze_extractor (bool, optional): 是否冻结特征提取层的参数。默认为 config.FREEZE_FEATURE_EXTRACTOR。
        aux_logits_user_preference (bool): 用户是否希望最终使用辅助分类器。
                                         GoogLeNet 预训练模型默认启用。
    返回:
        torch.nn.Module: 配置好的 GoogLeNet 模型。
    """
    # 使用配置文件中的默认值（如果未提供参数）
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if pretrained is None:
        pretrained = config.USE_PRETRAINED
    if freeze_extractor is None:
        freeze_extractor = config.FREEZE_FEATURE_EXTRACTOR

    # 加载 GoogLeNet 模型
    if pretrained:
        weights = models.GoogLeNet_Weights.IMAGENET1K_V1
        # 加载预训练模型时，始终初始以 aux_logits=True 加载，因为权重期望它们存在。
        model = models.googlenet(weights=weights, aux_logits=True)
        print(f"使用预训练的 GoogLeNet (ImageNet) 权重。初始加载时辅助分类器: 启用")

        # 现在，如果用户的偏好是禁用它们，则在加载后禁用。
        if not aux_logits_user_preference:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
            print("用户选择禁用辅助分类器，已在加载后移除。")
        else:
            print(f"用户选择使用辅助分类器。")

    else:  # 非预训练
        # 注意：如果不使用预训练权重，aux_logits 选项在模型初始化时可能行为不同，
        # torchvision 0.13+ 后，未预训练的 googlenet aux_logits 默认为 False，除非显式传入。
        # 为了保持一致性，我们传递 aux_logits_user_preference。
        model = models.googlenet(weights=None, num_classes=num_classes, aux_logits=aux_logits_user_preference)
        print(f"从头开始训练 GoogLeNet。辅助分类器: {'启用' if aux_logits_user_preference else '禁用'}")

    # 如果使用预训练模型，则需要修改分类头并可能冻结层
    if pretrained:
        # 冻结特征提取层的参数
        if freeze_extractor:
            print("冻结特征提取层的参数。")
            for param in model.parameters():
                param.requires_grad = False
        else:
            print("所有模型参数都将参与训练（微调模式）。")

        # 替换主分类头 (fc layer)
        num_ftrs_fc = model.fc.in_features
        model.fc = nn.Linear(num_ftrs_fc, num_classes)
        # 新的分类头参数默认 requires_grad=True

        # 仅当辅助分类器存在且用户希望使用时，才修改它们
        if aux_logits_user_preference and model.aux1 is not None and model.aux2 is not None:
            # 对于 torchvision >= 0.13 和 GoogLeNet_Weights.IMAGENET1K_V1，辅助分类器层名为 fc2
            # 但为了兼容性，检查是否存在 fc2 或 fc
            if hasattr(model.aux1, 'fc2'):
                num_ftrs_aux1 = model.aux1.fc2.in_features
                model.aux1.fc2 = nn.Linear(num_ftrs_aux1, num_classes)
            elif hasattr(model.aux1, 'fc'):  # 针对旧版本或不同权重结构的备选方案
                num_ftrs_aux1 = model.aux1.fc.in_features
                model.aux1.fc = nn.Linear(num_ftrs_aux1, num_classes)

            if hasattr(model.aux2, 'fc2'):
                num_ftrs_aux2 = model.aux2.fc2.in_features
                model.aux2.fc2 = nn.Linear(num_ftrs_aux2, num_classes)
            elif hasattr(model.aux2, 'fc'):
                num_ftrs_aux2 = model.aux2.fc.in_features
                model.aux2.fc = nn.Linear(num_ftrs_aux2, num_classes)

        # 如果 freeze_extractor 为 True，确保新添加/修改的层是可训练的
        if freeze_extractor:
            for param in model.fc.parameters():  # 确保主分类器可训练
                param.requires_grad = True
            if aux_logits_user_preference and model.aux1 is not None:  # 检查 model.aux1 是否实际存在
                for param in model.aux1.parameters():
                    param.requires_grad = True
            if aux_logits_user_preference and model.aux2 is not None:  # 检查 model.aux2 是否实际存在
                for param in model.aux2.parameters():
                    param.requires_grad = True

    # 如果不是预训练模型，模型已经直接使用 num_classes 和 aux_logits_user_preference 设置好了。
    # 对于非预训练模型，无需显式处理 freeze_extractor=True，因为所有参数都是新的。

    return model


# --------------------------------------------------------------------------------
# 测试函数 (Test function)
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    print("测试 GoogLeNet 模型创建...")

    print("\n--- 1. 测试预训练模型，冻结特征提取层 (迁移学习) ---")
    model_transfer = get_googlenet_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_extractor=True,
        aux_logits_user_preference=True  # 迁移学习时通常也使用辅助头
    )
    print(f"模型加载到: {next(model_transfer.parameters()).device}")
    print("检查参数的 requires_grad 状态 (仅显示可训练的分类器部分):")
    trainable_param_names_found = False
    for name, param in model_transfer.named_parameters():
        # 检查主分类器和辅助分类器（如果存在且用户选择使用）
        if param.requires_grad and ("fc" in name or ("aux" in name and model_transfer.aux1 is not None)):
            print(f"  可训练 (Trainable): {name} - {param.shape}")
            trainable_param_names_found = True
    if not trainable_param_names_found:
        print("  未找到明确设置为可训练的分类器参数（请检查逻辑）。")
    print("...")  # 代表可能有很多冻结的层未打印
    num_trainable_params_transfer = sum(p.numel() for p in model_transfer.parameters() if p.requires_grad)
    print(f"迁移学习模式下的可训练参数数量: {num_trainable_params_transfer}")

    print("\n--- 2. 测试预训练模型，微调所有层 ---")
    model_finetune = get_googlenet_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_extractor=False,
        aux_logits_user_preference=True
    )
    print(f"模型加载到: {next(model_finetune.parameters()).device}")
    all_params_trainable = all(p.requires_grad for p in model_finetune.parameters())
    print(f"微调模式下是否所有参数都可训练: {all_params_trainable}")
    num_trainable_params_finetune = sum(p.numel() for p in model_finetune.parameters() if p.requires_grad)
    print(f"微调模式下的可训练参数数量: {num_trainable_params_finetune}")

    # 模拟将模型移到配置的设备
    device = config.DEVICE
    if device == torch.device('cuda') and not torch.cuda.is_available():
        print(f"\n警告: config.DEVICE 设置为 'cuda' 但 CUDA 不可用。模型将保留在 CPU。")
        actual_device = torch.device('cpu')
    else:
        actual_device = device

    model_transfer.to(actual_device)
    print(f"\n模型已移至设备: {next(model_transfer.parameters()).device}")
    # 是的，我们可以处理这个
    # UserWarning。
     # 这个警告

    # 是
    # torchvision
    # 库发出的，旨在提醒你：虽然
    # GoogLeNet
    # 的主干网络权重是在
    # ImageNet
    # 上预训练的，但其辅助分类器（auxiliary
    # heads）的权重并没有像主干那样经过预训练，因此在使用它们时需要确保对它们进行训练。
    #
    # 由于你已经替换了辅助分类器的最后几层以适应你的
    # NUM_CLASSES，并且这些新层本身就需要训练，所以这个警告在你的情况下更多的是一个提示信息，而不是一个严重的问题。