# configs/config.py
# 全局配置类，管理超参数和文件路径

class Config:
    def __init__(self):
        # 图像参数
        self.IMG_WIDTH, self.IMG_HEIGHT = 299, 299  # Inception V3 输入尺寸
        self.BATCH_SIZE = 32  # 批量大小
        self.NUM_CLASSES = 5  # 花卉类别数
        # 训练参数
        self.EPOCHS_INITIAL = 15  # 初始训练轮次
        self.EPOCHS_FINE_TUNE = 15  # 微调轮次
        self.INITIAL_LR = 1e-3  # 初始训练学习率
        self.FINE_TUNE_LR = 1e-5  # 微调学习率
        self.OPTIMIZED_DROPOUT_RATE = 0.6  # 丢弃率
        self.OPTIMIZED_WEIGHT_DECAY = 1e-4  # 权重衰减
        # 数据和保存路径
        self.BASE_DIR = './flower_photos'  # 数据集根目录
        self.TRAIN_DIR = f'{self.BASE_DIR}/train'  # 训练集路径
        self.VALID_DIR = f'{self.BASE_DIR}/validation'  # 验证集路径
        self.TEST_DIR = f'{self.BASE_DIR}/test'  # 测试集路径
        self.MODEL_SAVE_PATH = 'best_flower_classifier_final_optimized.pth'  # 模型保存路径
        self.HISTORY_SAVE_PATH = 'training_history_final_optimized.json'  # 历史记录保存路径
        self.CSV_LOG_PATH = 'training_log_final_optimized.csv'  # 训练日志保存路径
