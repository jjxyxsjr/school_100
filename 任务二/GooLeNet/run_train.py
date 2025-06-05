# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
from 任务二.GooLeNet import config, data, model
from trainer import Trainer
import utils


def main():
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(f"======== 实验配置 ========")
    print(f"使用设备: {config.DEVICE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"权重衰减: {config.WEIGHT_DECAY}")
    print(f"阶段一 Epochs: {config.STAGE1_EPOCHS}, 学习率: {config.STAGE1_LR}")
    print(f"阶段二 Epochs: {config.STAGE2_EPOCHS}, 学习率: {config.STAGE2_LR}")
    print(f"==========================")

    # 1. 数据加载
    print("\n[1/4] 正在加载和划分数据...")
    train_loader1, train_loader2, val_loader, test_loader = data.get_dataloaders()

    # 2. 模型加载
    print("\n[2/4] 正在加载预训练的 AlexNet 模型...")
    alexnet_model = model.get_alexnet()
    print("模型加载完成。")

    # 3. 初始化训练器
    trainer = Trainer(model=alexnet_model, device=config.DEVICE)

    # --- 阶段一：冻结训练 ---
    print("\n[3/4] 开始执行第一阶段训练...")
    trainer.train_stage1(train_loader1, val_loader)

    # 在第一阶段结束后进行可视化
    utils.plot_stage1_curves(trainer.history)

    # 在测试集上评估第一阶段模型
    trainer.test(test_loader, "第一阶段")

    # --- 阶段二：微调训练 ---
    print("\n[4/4] 开始执行第二阶段训练...")
    trainer.train_stage2(train_loader2, val_loader)

    # 在第二阶段结束后进行最终可视化
    utils.plot_final_curves(trainer.history)

    # 在测试集上评估第二阶段模型
    trainer.test(test_loader, "第二阶段（最终）")

    print("\n--- 任务完成 ---")


if __name__ == '__main__':
    main()