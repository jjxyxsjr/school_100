"""
主模块：协调花卉分类任务的执行流程。
- 初始化模型、数据加载器、训练器、TTA 测试器和可视化工具。
- 执行模型训练、测试和结果可视化。
"""
# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import FlowerClassifier
from data_loader import FlowerDataLoader
from trainer import ModelTrainer
from tta import TTATester
from visualizer import TrainingVisualizer


if __name__ == '__main__':
    # 设置超参数
    data_dir = 'flower_photos'
    num_classes = 5
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001

    # 加载数据
    data_loader = FlowerDataLoader(data_dir, batch_size)
    image_datasets, dataloaders = data_loader.load_data()

    # 初始化模型
    print("加载预训练 AlexNet 模型...")
    model = FlowerClassifier(num_classes)
    print("模型加载和修改完成！特征提取层已冻结。")

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, scheduler, num_epochs)
    trained_model, history = trainer.train()

    # 使用 TTA 评估测试集
    tta_tester = TTATester(trained_model, image_datasets['test'])
    tta_tester.test()

    # 保存模型
    torch.save(trained_model.state_dict(), 'alexnet_flowers_final_tta.pth')
    print("\n最佳模型已保存至 alexnet_flowers_final_tta.pth")

    # 可视化训练结果
    print("生成训练曲线图...")
    visualizer = TrainingVisualizer()
    visualizer.visualize_training_curves(history, num_epochs)