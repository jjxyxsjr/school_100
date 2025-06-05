"""
主程序，协调数据处理、模型构建、训练和TTA测试。
初始化各模块，执行图像分类任务的完整流程。
"""

import torch
from data_handler import DataHandler
from model_manager import ModelManager
from trainer import Trainer
from tta_tester import TTATester


if __name__ == '__main__':
    # 设置超参数
    data_dir = 'flower_photos'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备是: {device} ✨")

    # 初始化数据处理
    data_handler = DataHandler(data_dir, batch_size)
    image_datasets, dataloaders, test_loader, dataset_sizes, class_names = data_handler.load_data()
    num_classes = len(class_names)
    print("数据加载完成！")
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['validation']}, 测试集大小: {dataset_sizes['test']}")
    print(f"类别数量: {num_classes}, 类别名称: {class_names}")

    # 初始化模型
    model_manager = ModelManager(num_classes, device, learning_rate)
    print("模型构建完成，已替换分类头并冻结主干网络。")

    # 训练模型
    trainer = Trainer(model_manager, dataloaders, dataset_sizes, num_epochs, device)
    trained_model, history = trainer.train()
    print("\n正在绘制训练曲线图...")
    trainer.plot_curves()

    # 测试模型（使用TTA）
    tta_tester = TTATester(trained_model, test_loader, data_handler.tta_transforms, device)
    tta_tester.test()