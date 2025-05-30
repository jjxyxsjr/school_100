# alexnet_project/config.py

import torch

# --- 1. 路径和文件名 ---
DATASET_PATH = r'D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\data_new'
JSON_OUTPUT_FILE = 'training_metrics_oop.json'
BEST_MODEL_PATH = 'best_model_oop.pth'
# # 为新实验修改文件名
# JSON_OUTPUT_FILE = 'training_run_2.json'
# BEST_MODEL_PATH = 'best_model_run_2.pth'
# --- 2. 模型与数据参数 ---
NUM_CLASSES = 102
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 3. 训练超参数 ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50
PATIENCE = 5  # 早停机制的“耐心值”