"""
Main entry point for the flower classification task using EfficientNet-B0.
Coordinates data loading, model building, training, visualization, and TTA testing.
All components are modularized and imported from other files in the same directory.
"""

import torch
from data_loader import DataLoaderFactory
from model_builder import ModelBuilder
from trainer import ModelTrainer
from visualizer import Visualizer
from tta_tester import TTATester

def main():
    # 设置超参数和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备是: {device} ✨")
    data_dir = 'flower_photos'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 1. 数据加载
    data_loader_factory = DataLoaderFactory(data_dir, batch_size)
    dataloaders, dataset_sizes, class_names, tta_transforms = data_loader_factory.create_dataloaders()
    num_classes = len(class_names)
    print("数据加载完成！")
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['validation']}, 测试集大小: {dataset_sizes['test']}")
    print(f"类别数量: {num_classes}, 类别名称: {class_names}")

    # 2. 构建模型
    model_builder = ModelBuilder(num_classes, learning_rate, device)
    model, criterion, optimizer = model_builder.build_efficientnet()

    # 3. 训练模型
    trainer = ModelTrainer(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, device)
    trained_model, history = trainer.train()

    # 4. 绘制训练曲线
    visualizer = Visualizer()
    visualizer.plot_curves(history)

    # 5. 测试时增强（TTA）
    tta_tester = TTATester(trained_model, dataloaders['test'], tta_transforms, device)
    tta_tester.test_with_tta()

if __name__ == '__main__':
    main()