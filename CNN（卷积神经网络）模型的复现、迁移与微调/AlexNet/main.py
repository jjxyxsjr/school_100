# alexnet_project/main.py

import config
from loader import get_data_loaders
from alexnet_model import create_alexnet_model
from trainer import ModelTrainer


def main():
    # 1. 打印配置信息
    print(f"使用的设备: {config.DEVICE}")
    print(f"开始训练，总周期数: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    # 2. 加载数据
    dataloaders, dataset_sizes = get_data_loaders(config.DATASET_PATH, config.BATCH_SIZE)
    if dataloaders is None:
        return

    # 3. 创建模型
    model = create_alexnet_model(num_classes=config.NUM_CLASSES)

    # 4. 创建训练器并开始训练
    trainer = ModelTrainer(model, dataloaders, dataset_sizes, config)
    history = trainer.train()



if __name__ == '__main__':
    main()