# flowers_transfer/config.py

import torch
import os

# --------------------------------------------------------------------------------
# 基本配置 (Basic Configuration)
# --------------------------------------------------------------------------------
PROJECT_NAME = "GoogLeNet_Flowers102_Transfer"
RANDOM_SEED = 42  # 随机种子，用于复现实验结果
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择 GPU 或 CPU

# --------------------------------------------------------------------------------
# 数据集配置 (Dataset Configuration)
# --------------------------------------------------------------------------------
# 用户提供的实际路径
FLOWERS102_BASE_DATA_DIR = r"D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\GooLeNet\flowers_transfer\data"

IMAGE_DIR = os.path.join(FLOWERS102_BASE_DATA_DIR, '102flowers', 'jpg') # 图片文件夹路径
LABEL_MAT_FILE = os.path.join(FLOWERS102_BASE_DATA_DIR, 'imagelabels.mat') # 标签 .mat 文件路径
SETID_MAT_FILE = os.path.join(FLOWERS102_BASE_DATA_DIR, 'setid.mat') # 数据集划分 .mat 文件路径

# 图像预处理参数
IMAGE_SIZE = 224  # GoogLeNet 输入图像尺寸
NORM_MEAN = [0.485, 0.456, 0.406] # ImageNet 均值
NORM_STD = [0.229, 0.224, 0.225]  # ImageNet 标准差

NUM_CLASSES = 102  # Flowers102 数据集的类别数量
NUM_WORKERS = 4    # DataLoader 使用的子进程数量，根据你的 CPU核心数调整

# --------------------------------------------------------------------------------
# 模型配置 (Model Configuration)
# --------------------------------------------------------------------------------
MODEL_NAME = "googlenet"
USE_PRETRAINED = True  # 是否使用在 ImageNet 上预训练的 GoogLeNet权重
FREEZE_FEATURE_EXTRACTOR = True # 在迁移学习初期，通常冻结特征提取层
                               # 设置为 False 进行微调 (fine-tuning)

# --------------------------------------------------------------------------------
# 训练配置 (Training Configuration)
# --------------------------------------------------------------------------------
# 对于特征提取 (FREEZE_FEATURE_EXTRACTOR = True):
TRANSFER_LEARNING_EPOCHS = 10
TRANSFER_LEARNING_BATCH_SIZE = 32
TRANSFER_LEARNING_LR = 1e-3

# 对于微调 (FREEZE_FEATURE_EXTRACTOR = False):
FINETUNE_EPOCHS = 10
FINETUNE_BATCH_SIZE = 16
FINETUNE_LR_BACKBONE = 1e-5
FINETUNE_LR_CLASSIFIER = 1e-4

# 通用优化器和调度器参数
OPTIMIZER_TYPE = "Adam"  # 可选 "SGD", "Adam", "AdamW"
WEIGHT_DECAY = 1e-4

# 学习率调度器 (Learning Rate Scheduler)
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "StepLR"
STEP_LR_STEP_SIZE = 7
STEP_LR_GAMMA = 0.1

# --------------------------------------------------------------------------------
# 输出配置 (Output Configuration)
# --------------------------------------------------------------------------------
# 获取 config.py 文件所在的目录 (应该是 flowers_transfer)
_current_file_dir = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(_current_file_dir, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 确保输出目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# 辅助函数或打印配置 (Optional: Helper function or print config)
# --------------------------------------------------------------------------------
def print_config():
    """打印所有配置项"""
    print("-------------------- Configuration --------------------")
    for key, value in globals().items():
        if not key.startswith("_") and key.isupper(): # 只打印大写常量 (不包括以_开头的)
            print(f"{key}: {value}")
    print("-------------------------------------------------------")

if __name__ == '__main__':
    print_config()
    print(f"\nResolved IMAGE_DIR: {IMAGE_DIR}")
    print(f"Resolved LABEL_MAT_FILE: {LABEL_MAT_FILE}")
    print(f"Resolved SETID_MAT_FILE: {SETID_MAT_FILE}")
    print(f"Resolved OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")