# main.py
from torchvision import transforms
from trainer import CatDogTrainer

# 图像预处理流程
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 建议加上标准化
])

# 训练和测试数据路径
train_dir = "./data/train"
test_dir = "./data/test"

if __name__ == "__main__":
    # 设置两个阶段的训练轮数
    TRANSFER_LEARNING_EPOCHS = 5  # 第一阶段（迁移学习）的轮数
    FINE_TUNE_EPOCHS = 5        # 第二阶段（微调）的轮数
    FINE_TUNE_LR = 0.0001        # 微调阶段的学习率

    trainer = CatDogTrainer(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        epochs=TRANSFER_LEARNING_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        fine_tune_lr=FINE_TUNE_LR
    )
    trainer.run()