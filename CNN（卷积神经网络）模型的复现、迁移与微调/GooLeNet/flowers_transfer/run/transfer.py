# flowers_transfer/run/transfer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import os
import sys
from tqdm import tqdm # <--- 添加这一行
# 将项目根目录（GooLeNet/）添加到Python路径中，以便导入模块
# 假设 run/ 目录在 flowers_transfer/ 内部，而 flowers_transfer 在 GooLeNet/ 内部
# 或者，更常见的是，你从 GooLeNet/ 目录运行 python flowers_transfer/run/transfer.py
# 在这种情况下，flowers_transfer 应该可以直接作为包导入。
# 为确保在各种运行方式下都能工作，我们可以动态添加路径：
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # run/ 目录
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))  # 推测 GooLeNet/
# 如果 flowers_transfer 是直接的父目录 (即脚本在 flowers_transfer/run/ 下)
FLOWER_TRANSFER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# 添加 flowers_transfer 的父目录（通常是项目根目录）到 sys.path
# 这样就可以 import flowers_transfer.config 等
# 如果你的 PyCharm 工作目录设置正确，或者你从项目根目录运行，这可能不是必需的
if FLOWER_TRANSFER_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(FLOWER_TRANSFER_ROOT))  # 添加 flowers_transfer 的父目录
if FLOWER_TRANSFER_ROOT not in sys.path:  # 确保 flowers_transfer 目录本身也在路径中，以便其内部模块可以相互导入
    sys.path.insert(0, FLOWER_TRANSFER_ROOT)

from flowers_transfer import config
from flowers_transfer.dataset import get_dataloaders
from flowers_transfer.model import get_googlenet_model
from flowers_transfer.trainer import Trainer
from flowers_transfer.utils import utils as project_utils  # 重命名以避免与标准库 utils 冲突
from flowers_transfer.utils import visualizer


def run_transfer_learning():
    """执行迁移学习的主要流程。"""
    print("开始迁移学习流程...")

    # 1. 设置随机种子
    project_utils.set_seed(config.RANDOM_SEED)

    # 2. 打印配置（可选）
    config.print_config()
    print(f"当前设备: {config.DEVICE}")

    # 3. 获取数据加载器
    print("正在加载数据...")
    # 对于迁移学习，使用 TRANSFER_LEARNING_BATCH_SIZE
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size_train=config.TRANSFER_LEARNING_BATCH_SIZE,
        batch_size_eval=config.TRANSFER_LEARNING_BATCH_SIZE  # 验证和测试通常使用相同的批大小
    )
    print("数据加载完成。")

    # 4. 获取模型
    print("正在初始化模型 (迁移学习模式)...")
    # 对于迁移学习，明确设置 pretrained=True, freeze_extractor=True
    # aux_logits_user_preference 可以从 config 读取，或在此处硬编码
    model = get_googlenet_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_extractor=True,  # 关键：迁移学习时冻结特征提取器
        aux_logits_user_preference=True  # GoogLeNet 训练时通常使用辅助损失
    )
    model.to(config.DEVICE)  # 将模型移至设备

    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 6. 定义优化器
    # *只*优化那些 requires_grad = True 的参数
    # 对于迁移学习（冻结模式），这通常只是新分类头的参数
    print("正在配置优化器 (仅训练分类头)...")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if config.OPTIMIZER_TYPE.lower() == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=config.TRANSFER_LEARNING_LR,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER_TYPE.lower() == "sgd":
        optimizer = optim.SGD(
            trainable_params,
            lr=config.TRANSFER_LEARNING_LR,
            momentum=0.9,  # SGD 通常需要 momentum
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER_TYPE.lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_params,
            lr=config.TRANSFER_LEARNING_LR,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"不支持的优化器类型: {config.OPTIMIZER_TYPE}")

    print(f"优化器: {config.OPTIMIZER_TYPE}, 初始学习率: {config.TRANSFER_LEARNING_LR}")

    # 7. 定义学习率调度器 (可选)
    scheduler = None
    if config.USE_LR_SCHEDULER:
        if config.LR_SCHEDULER_TYPE.lower() == "steplr":
            scheduler = StepLR(optimizer, step_size=config.STEP_LR_STEP_SIZE, gamma=config.STEP_LR_GAMMA)
            print(f"使用 StepLR 调度器: step_size={config.STEP_LR_STEP_SIZE}, gamma={config.STEP_LR_GAMMA}")
        elif config.LR_SCHEDULER_TYPE.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            print("使用 ReduceLROnPlateau 调度器: factor=0.1, patience=5")
        elif config.LR_SCHEDULER_TYPE.lower() == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, T_max=config.TRANSFER_LEARNING_EPOCHS,
                                          eta_min=1e-6)  # T_max 通常是总 epochs
            print(f"使用 CosineAnnealingLR 调度器: T_max={config.TRANSFER_LEARNING_EPOCHS}")
        # 可以添加更多调度器类型

    # 8. 实例化 Trainer
    print("正在实例化 Trainer...")
    trainer_instance = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.TRANSFER_LEARNING_EPOCHS,
        checkpoint_dir=config.CHECKPOINT_DIR,
        model_name="googlenet_flowers102_transfer_best.pth"  # 特定于此运行的名称
    )

    # 9. 开始训练
    print("开始训练模型...")
    history = trainer_instance.train()

    # 10. （可选）在测试集上评估最终（或最佳）模型
    print("\n训练完成。正在测试集上评估最佳模型...")
    # 加载最佳模型权重
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "googlenet_flowers102_transfer_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已从 {best_model_path} 加载最佳模型权重进行最终评估。")
    else:
        print(f"警告: 未找到最佳模型 {best_model_path}，将使用当前模型状态进行评估。")

    model.eval()
    test_loss = 0
    test_corrects = 0
    test_total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data).item()
            test_total_samples += labels.size(0)

    final_test_loss = test_loss / test_total_samples
    final_test_acc = test_corrects / test_total_samples
    print(f"最终测试集评估: 损失 = {final_test_loss:.4f}, 准确率 = {final_test_acc:.4f}")
    history['test_loss'] = final_test_loss  # 将测试结果也添加到 history 中
    history['test_acc'] = final_test_acc

    # 11. 可视化训练历史
    print("正在生成训练曲线图...")
    visualizer.plot_training_history(history, plot_name="transfer_learning_curves.png")

    print("迁移学习流程结束。")


if __name__ == '__main__':
    run_transfer_learning()