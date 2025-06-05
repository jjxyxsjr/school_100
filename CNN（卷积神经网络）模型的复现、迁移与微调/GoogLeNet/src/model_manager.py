# src/model_manager.py
# 管理模型初始化、修改和设备分配

import torch  # 用于设备管理和模型操作
import torch.nn as nn  # 提供神经网络组件
from torchvision import models  # 提供预训练模型

class ModelManager:
    def __init__(self, config, device):
        """初始化模型管理器，构建并配置模型"""
        self.config = config  # 存储配置对象，包含超参数
        self.device = device  # 存储计算设备（GPU 或 CPU）
        self.model = self._build_model()  # 构建并配置模型

    def _build_model(self):
        """构建并配置 Inception V3 模型"""
        # 加载预训练 Inception V3
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)  # 使用 ImageNet 预训练权重
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False  # 禁用所有参数的梯度计算，防止初始训练更新预训练权重

        # 修改主分类器
        num_features_main = model.fc.in_features  # 获取主分类器输入特征数（2048） # type: ignore
        model.fc = nn.Sequential(
            nn.Linear(num_features_main, 1024),  # 将 2048 维特征映射到 1024 维
            nn.ReLU(),  # 添加 ReLU 激活函数，增加非线性
            nn.Dropout(p=self.config.OPTIMIZED_DROPOUT_RATE),  # 添加丢弃层（丢弃率 0.6），防止过拟合
            nn.Linear(1024, self.config.NUM_CLASSES)  # 将 1024 维映射到 5 个类别（花卉分类）
        )

        # 修改辅助分类器（如果存在）
        if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
            num_features_aux = model.AuxLogits.fc.in_features  # 获取辅助分类器输入特征数 # type: ignore
            model.AuxLogits.fc = nn.Sequential(
                nn.Linear(num_features_aux, 1024),  # 将输入特征映射到 1024 维
                nn.ReLU(),  # 添加 ReLU 激活函数
                nn.Dropout(p=self.config.OPTIMIZED_DROPOUT_RATE),  # 添加丢弃层（丢弃率 0.6）
                nn.Linear(1024, self.config.NUM_CLASSES)  # 映射到 5 个类别
            )
        return model.to(self.device)  # 将模型移动到指定设备（GPU 或 CPU）

    def prepare_for_fine_tuning(self, fine_tune_after_layer='Mixed_7a'):
        """为微调解冻指定层及后续层"""
        found_fine_tune_layer = False
        for name, child in self.model.named_children():
            if fine_tune_after_layer in name:
                found_fine_tune_layer = True  # 找到指定层（Mixed_7a），标记开始解冻
            if found_fine_tune_layer:
                for param in child.parameters():
                    param.requires_grad = True  # 解冻 Mixed_7a 及后续层（如 Mixed_7b、Mixed_7c、fc）的参数，允许微调
        # 确保分类器层可训练
        for param in self.model.fc.parameters():
            param.requires_grad = True  # 显式设置主分类器（fc）参数可训练，确保输出层始终可更新
        if hasattr(self.model, 'AuxLogits') and self.model.AuxLogits is not None:
            for param in self.model.AuxLogits.fc.parameters():
                param.requires_grad = True  # 显式设置辅助分类器（AuxLogits.fc）参数可训练，确保辅助输出可更新