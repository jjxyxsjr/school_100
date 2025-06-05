# config.py

import torch

# --- 数据集和路径配置 ---
ORIGIN_DATA_PATH = 'flower_photos'
DATA_ROOT = 'data_split'

# --- 模型和设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
MODEL_PATH_STAGE1 = 'best_model_stage1.pth'
MODEL_PATH_STAGE2 = 'best_model_stage2.pth'

# 💡 新增: 早停的耐心值
# 如果验证损失连续 3 轮没有下降，则提前停止训练
PATIENCE = 3

# --- 训练参数 ---
BATCH_SIZE = 16 # 使用之前确认能运行的较小批次大小

# --- 第一阶段：冻结训练 ---
EPOCHS_STAGE1 = 15
LR_STAGE1 = 1e-3
WEIGHT_DECAY_STAGE1 = 1e-4

# --- 第二阶段：微调 ---
EPOCHS_STAGE2 = 10
LR_STAGE2 = 2e-5
WEIGHT_DECAY_STAGE2 = 1e-4