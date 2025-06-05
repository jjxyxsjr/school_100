# run_train.py

import config
from data import split_dataset, get_dataloaders
from model import get_model, unfreeze_model_layers
from trainer import Trainer


def main():
    """主执行函数"""
    print("项目启动...")
    print(f"使用的设备是: {config.DEVICE}")

    # 1. 划分数据集 (如果尚未划分)
    split_dataset()

    # 2. 获取数据加载器
    dataloaders, dataset_sizes, class_names = get_dataloaders()
    print(
        f"数据加载完毕。训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['validation']}, 测试集大小: {dataset_sizes['test']}")

    # 3. 加载模型
    model = get_model(config.DEVICE)

    # 4. 初始化训练器
    trainer = Trainer(model, dataloaders, dataset_sizes, config.DEVICE)

    # --- 阶段一：冻结训练 ---
    trainer.train(
        stage_num=1,
        epochs=config.EPOCHS_STAGE1,
        lr=config.LR_STAGE1,
        weight_decay=config.WEIGHT_DECAY_STAGE1,
        model_save_path=config.MODEL_SAVE_PATH_STAGE1,
        visualization_path=config.VISUALIZATION_PATH_STAGE1
    )

    # 在测试集上评估第一阶段的模型
    print("\n--- 评估第一阶段模型在测试集上的表现 ---")
    trainer.test()

    # --- 阶段二：解冻微调 ---
    # 解冻模型的部分层
    unfreeze_model_layers(model)

    # 用更新后的模型（解冻）重新初始化训练器
    # 或者直接在现有 trainer 实例上继续训练
    trainer.model = model  # 确保 trainer 使用的是解冻后的模型

    trainer.train(
        stage_num=2,
        epochs=config.EPOCHS_STAGE2,
        lr=config.LR_STAGE2,
        weight_decay=config.WEIGHT_DECAY_STAGE2,
        model_save_path=config.MODEL_SAVE_PATH_STAGE2,
        visualization_path=config.VISUALIZATION_PATH_STAGE2,
        load_path=config.MODEL_SAVE_PATH_STAGE1  # 加载第一阶段的最佳模型
    )

    # 在测试集上评估第二阶段（最终）的模型
    print("\n--- 评估第二阶段模型在测试集上的最终表现 ---")
    trainer.test()

    print("\n项目运行结束。")


if __name__ == '__main__':
    main()