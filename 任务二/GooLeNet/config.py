# -*- coding: utf-8 -*-
import torch
import os

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 现在直接指向包含所有类别文件夹的根目录
DATA_DIR = os.path.join(BASE_DIR, "flower_photos")

# --- 设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 模型与数据配置 ---
NUM_CLASSES = 5
# 新的数据划分比例
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
# 训练集内部划分比例
TRAIN_SPLIT_RATIO = 0.7

# --- 训练超参数 ---
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4 # 权重衰减

# --- 阶段一：冻结训练 ---
STAGE1_EPOCHS = 15 # 可以适当增加Epoch数，因为有了更可靠的验证集
STAGE1_LR = 1e-3
PATIENCE = 3
STAGE1_MODEL_PATH = os.path.join(BASE_DIR, "best_model_stage1.pth")

# --- 阶段二：微调训练 ---
STAGE2_EPOCHS = 10 # 微调阶段也可以适当增加
STAGE2_LR = 5e-6   # 使用建议的更低的学习率
STAGE2_MODEL_PATH = os.path.join(BASE_DIR, "best_model_stage2.pth")

# --- 图像变换参数 ---
IMAGE_SIZE = 224