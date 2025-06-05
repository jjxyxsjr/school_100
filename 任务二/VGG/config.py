# config.py

import torch

# --- 数据集参数 ---
# 原始数据集路径
ORIG_DATA_PATH = 'flower_photos'
# 新划分的数据集路径
BASE_PATH = 'data_split'
# 训练、验证、测试集划分比例
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1
# 类别数量
NUM_CLASSES = 5
# 批处理大小
BATCH_SIZE = 32

# --- 模型和设备参数 ---
# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ImageNet 图像均值和标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --- 训练阶段一参数: 冻结训练 ---
# 学习率
LR_STAGE1 = 1e-3
# 权重衰减 (L2 正则化)
WEIGHT_DECAY_STAGE1 = 1e-4
# 训练轮数
EPOCHS_STAGE1 = 15
# Early Stopping 的耐心值
EARLY_STOPPING_PATIENCE = 3
# 最佳模型保存路径
MODEL_SAVE_PATH_STAGE1 = 'best_model_stage1.pth'
# 可视化结果保存路径
VISUALIZATION_PATH_STAGE1 = 'training_results_stage1.png'


# --- 训练阶段二参数: 微调 ---
# 学习率 (一个较小的值)
LR_STAGE2 = 5e-5
# 权重衰减
WEIGHT_DECAY_STAGE2 = 1e-4
# 训练轮数
EPOCHS_STAGE2 = 10
# 最佳模型保存路径
MODEL_SAVE_PATH_STAGE2 = 'best_model_stage2.pth'
# 可视化结果保存路径
VISUALIZATION_PATH_STAGE2 = 'training_results_stage2.png'