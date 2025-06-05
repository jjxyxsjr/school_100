# main.py
# 主脚本，协调各模块执行花卉分类任务

import torch
import multiprocessing
import os
from configs.config import Config
from src.data_manager import DataManager
from src.model_manager import ModelManager
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 初始化配置和设备
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 数据准备
    print("正在准备数据...")
    data_manager = DataManager(config)
    data_manager.load_data()

    # 模型初始化
    print("\n正在构建模型...")
    model_manager = ModelManager(config, device)

    # 初始训练
    print("\n开始初始训练 (特征提取)...")
    trainer = Trainer(config, model_manager.model, data_manager.dataloaders, data_manager.dataset_sizes, device)
    model, train_acc_initial, val_acc_initial, train_loss_initial, val_loss_initial, best_val_acc_initial, stopped_epoch_initial = trainer.train(
        config.EPOCHS_INITIAL, config.INITIAL_LR, phase_name="初始训练", patience=5
    )

    # 微调
    print("\n准备进行微调...")
    model_manager.prepare_for_fine_tuning()
    print("\n开始微调...")
    model, train_acc_ft, val_acc_ft, train_loss_ft, val_loss_ft, best_val_acc_ft, stopped_epoch_ft = trainer.train(
        config.EPOCHS_FINE_TUNE, config.FINE_TUNE_LR, phase_name="微调", current_best_acc=best_val_acc_initial, patience=7
    )

    # 保存训练历史
    print("\n正在保存训练历史...")
    visualizer = Visualizer(config)
    combined_history = {
        'accuracy': train_acc_initial + train_acc_ft,
        'val_accuracy': val_acc_initial + val_acc_ft,
        'loss': train_loss_initial + train_loss_ft,
        'val_loss': val_loss_initial + val_loss_ft
    }
    visualizer.save_history(combined_history)

    # 测试集评估（TTA）
    if 'test' in data_manager.dataloaders:
        print("\n正在加载最佳模型并使用 TTA 进行最终评估...")
        model_manager.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        evaluator = Evaluator(config, model_manager.model, data_manager.dataloaders['test'], 
                             data_manager.dataset_sizes['test'], device)
        test_loss, test_acc = evaluator.evaluate_with_tta()
        print(f"\n最终测试集表现 (TTA):")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc * 100:.2f}%")

    # 可视化
    print("\n正在绘制训练历史图...")
    visualizer.plot_training_history(combined_history, stopped_epoch_initial + 1)

    print("\n面向对象版花卉分类项目执行完毕。")
