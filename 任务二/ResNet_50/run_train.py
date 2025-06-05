# run_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import config
import data
import model as model_loader
import trainer
import utils


def main():
    print(f"使用的设备是: {config.DEVICE}")

    print("\n--- [步骤 0/3] 准备数据集中... ---")
    data.split_dataset()

    print("\n加载数据中...")
    train_loader, val_loader, test_loader = data.get_dataloaders()
    print("数据集加载完成。")

    # --- 步骤 1: 第一阶段训练 ---
    print("\n--- [步骤 1/3] 开始第一阶段训练: 仅训练全连接层 ---")
    # 💡 修改: 调用 get_resnet50
    model_s1 = model_loader.get_resnet50(stage=1)

    params_to_update_s1 = [p for p in model_s1.parameters() if p.requires_grad]
    optimizer_s1 = optim.Adam(params_to_update_s1, lr=config.LR_STAGE1, weight_decay=config.WEIGHT_DECAY_STAGE1)
    criterion = nn.CrossEntropyLoss()

    history_s1 = trainer.train_and_evaluate(
        model=model_s1,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_s1,
        criterion=criterion,
        epochs=config.EPOCHS_STAGE1,
        device=config.DEVICE,
        model_save_path=config.MODEL_PATH_STAGE1,
        patience=config.PATIENCE
    )

    utils.plot_curves(history_s1, "stage1")

    print("\n在测试集上评估第一阶段的最佳模型...")
    # 💡 修改: 确保加载权重前使用正确的模型结构
    model_to_test_s1 = model_loader.get_resnet50(stage=1)
    model_to_test_s1.load_state_dict(torch.load(config.MODEL_PATH_STAGE1))
    test_acc_s1 = utils.test_model(model_to_test_s1, test_loader, config.DEVICE)
    print(f"✅ 第一阶段 -> 测试集准确率: {test_acc_s1:.2f}%")

    # --- 步骤 2: 第二阶段训练 ---
    print("\n\n--- [步骤 2/3] 开始第二阶段训练: 微调部分卷积层 ---")
    # 💡 修改: 调用 get_resnet50
    model_s2 = model_loader.get_resnet50(stage=2)
    model_s2.load_state_dict(torch.load(config.MODEL_PATH_STAGE1))
    print(f"成功加载第一阶段最佳权重 '{config.MODEL_PATH_STAGE1}'。")

    params_to_update_s2 = [p for p in model_s2.parameters() if p.requires_grad]
    optimizer_s2 = optim.Adam(params_to_update_s2, lr=config.LR_STAGE2, weight_decay=config.WEIGHT_DECAY_STAGE2)

    history_s2 = trainer.train_and_evaluate(
        model=model_s2,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_s2,
        criterion=criterion,
        epochs=config.EPOCHS_STAGE2,
        device=config.DEVICE,
        model_save_path=config.MODEL_PATH_STAGE2,
        patience=config.PATIENCE
    )

    utils.plot_curves(history_s2, "stage2")

    print("\n在测试集上评估第二阶段的最佳模型...")
    # 💡 修改: 确保加载权重前使用正确的模型结构
    model_to_test_s2 = model_loader.get_resnet50(stage=2)
    model_to_test_s2.load_state_dict(torch.load(config.MODEL_PATH_STAGE2))
    test_acc_s2 = utils.test_model(model_to_test_s2, test_loader, config.DEVICE)
    print(f"✅ 第二阶段 -> 测试集准确率: {test_acc_s2:.2f}%")

    print("\n\n--- [步骤 3/3] 训练流程结束，结果总结 ---")
    print(f"第一阶段最佳模型测试准确率: {test_acc_s1:.2f}% (保存于 {config.MODEL_PATH_STAGE1})")
    print(f"第二阶段最佳模型测试准确率: {test_acc_s2:.2f}% (保存于 {config.MODEL_PATH_STAGE2})")


if __name__ == '__main__':
    main()