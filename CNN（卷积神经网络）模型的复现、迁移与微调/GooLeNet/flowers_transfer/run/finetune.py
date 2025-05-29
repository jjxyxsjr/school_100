# flowers_transfer/run/finetune.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import os
import sys
from tqdm import tqdm

# 与 transfer.py 中类似的路径设置，确保模块可以被导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # run/ 目录
FLOWER_TRANSFER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

if os.path.dirname(FLOWER_TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(FLOWER_TRANSFER_ROOT))
if FLOWER_TRANSFER_ROOT not in sys.path:
    sys.path.insert(0, FLOWER_TRANSFER_ROOT)

from flowers_transfer import config
from flowers_transfer.dataset import get_dataloaders
from flowers_transfer.model import get_googlenet_model
from flowers_transfer.trainer import Trainer
from flowers_transfer.utils import utils as project_utils
from flowers_transfer.utils import visualizer


def run_finetuning():
    """执行微调的主要流程。"""
    print("开始微调流程...")

    # 1. 设置随机种子
    project_utils.set_seed(config.RANDOM_SEED)

    # 2. 打印配置（可选）
    # config.print_config() # 迁移学习时已打印，这里可以省略或按需开启
    print(f"当前设备: {config.DEVICE}")
    print(f"微调周期: {config.FINETUNE_EPOCHS}, 批大小: {config.FINETUNE_BATCH_SIZE}")
    print(f"学习率 (主干): {config.FINETUNE_LR_BACKBONE}, 学习率 (分类头): {config.FINETUNE_LR_CLASSIFIER}")

    # 3. 获取数据加载器
    print("正在加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size_train=config.FINETUNE_BATCH_SIZE,
        batch_size_eval=config.FINETUNE_BATCH_SIZE
    )
    print("数据加载完成。")

    # 4. 获取模型
    print("正在初始化模型 (微调模式)...")
    # 对于微调，pretrained=True (因为我们要用预训练的特征), freeze_extractor=False
    model = get_googlenet_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,  # 确保加载预训练结构
        freeze_extractor=False,  # 关键：微调时不冻结特征提取器
        aux_logits_user_preference=True  # 通常在微调时也使用辅助损失
    )

    # 4.1 (推荐) 加载迁移学习阶段训练好的模型权重作为起点
    transfer_model_path = os.path.join(config.CHECKPOINT_DIR, "googlenet_flowers102_transfer_best.pth")
    if os.path.exists(transfer_model_path):
        try:
            checkpoint = torch.load(transfer_model_path, map_location=config.DEVICE)
            # 加载状态字典，需要确保与当前模型结构兼容
            # 如果迁移学习时 aux_logits 设置不同，这里可能需要小心处理
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已从 {transfer_model_path} 加载迁移学习阶段的最佳模型权重。")
            print(f"迁移学习模型在验证集上的准确率为: {checkpoint.get('best_val_acc', 'N/A')}")
        except Exception as e:
            print(f"加载迁移学习模型权重失败: {e}。将使用ImageNet预训练权重进行微调。")
            # 如果加载失败，模型仍然是ImageNet预训练的（因为get_googlenet_model中pretrained=True）
    else:
        print(f"未找到迁移学习模型 {transfer_model_path}。将使用ImageNet预训练权重进行微调。")

    model.to(config.DEVICE)

    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 6. 定义优化器 (为不同部分设置不同学习率)
    print("正在配置优化器 (为骨干和分类头设置不同学习率)...")

    # 识别分类器参数 (fc, aux1, aux2)
    classifier_params_ids = set()
    if hasattr(model, 'fc'):
        classifier_params_ids.update(map(id, model.fc.parameters()))
    if model.aux_logits and hasattr(model, 'aux1') and model.aux1 is not None:
        classifier_params_ids.update(map(id, model.aux1.parameters()))
    if model.aux_logits and hasattr(model, 'aux2') and model.aux2 is not None:
        classifier_params_ids.update(map(id, model.aux2.parameters()))

    backbone_params = [p for p in model.parameters() if id(p) not in classifier_params_ids and p.requires_grad]
    # 需要确保分类器参数也是 requires_grad=True 的，get_googlenet_model 应该已经处理了
    # 但这里我们只筛选，不改变 requires_grad 状态
    classifier_params_list = [p for p in model.parameters() if id(p) in classifier_params_ids and p.requires_grad]

    # 检查是否有参数被遗漏
    all_params_in_optimizer = len(backbone_params) + len(classifier_params_list)
    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    if all_params_in_optimizer != total_model_params:
        print(
            f"警告: 优化器参数数量 ({all_params_in_optimizer}) 与模型可训练参数数量 ({total_model_params}) 不匹配。请检查参数分组逻辑。")

    param_groups = [
        {'params': backbone_params, 'lr': config.FINETUNE_LR_BACKBONE},
        {'params': classifier_params_list, 'lr': config.FINETUNE_LR_CLASSIFIER}
    ]

    # 移除空的参数组 (例如，如果分类器参数列表为空)
    param_groups = [pg for pg in param_groups if pg['params']]

    if config.OPTIMIZER_TYPE.lower() == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER_TYPE.lower() == "sgd":
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER_TYPE.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"不支持的优化器类型: {config.OPTIMIZER_TYPE}")

    print(f"优化器: {config.OPTIMIZER_TYPE}")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  参数组 {i}: 学习率 = {group['lr']}, 参数数量 = {len(group['params'])}")

    # 7. 定义学习率调度器 (可选)
    scheduler = None
    if config.USE_LR_SCHEDULER:
        if config.LR_SCHEDULER_TYPE.lower() == "steplr":
            scheduler = StepLR(optimizer, step_size=config.STEP_LR_STEP_SIZE, gamma=config.STEP_LR_GAMMA)
            print(f"使用 StepLR 调度器: step_size={config.STEP_LR_STEP_SIZE}, gamma={config.STEP_LR_GAMMA}")
        elif config.LR_SCHEDULER_TYPE.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                          verbose=True)  # 微调时 patience 可能小一些
            print("使用 ReduceLROnPlateau 调度器: factor=0.1, patience=3")
        elif config.LR_SCHEDULER_TYPE.lower() == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, T_max=config.FINETUNE_EPOCHS, eta_min=1e-7)  # eta_min 可以更小
            print(f"使用 CosineAnnealingLR 调度器: T_max={config.FINETUNE_EPOCHS}")

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
        epochs=config.FINETUNE_EPOCHS,
        checkpoint_dir=config.CHECKPOINT_DIR,
        model_name="googlenet_flowers102_finetune_best.pth"  # 微调模型的特定名称
    )

    # 9. 开始训练
    print("开始训练模型 (微调)...")
    history = trainer_instance.train()

    # 10. （可选）在测试集上评估最终（或最佳）模型
    print("\n微调完成。正在测试集上评估最佳模型...")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "googlenet_flowers102_finetune_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已从 {best_model_path} 加载最佳微调模型权重进行最终评估。")
    else:
        print(f"警告: 未找到最佳微调模型 {best_model_path}，将使用当前模型状态进行评估。")

    model.eval()
    test_loss = 0
    test_corrects = 0
    test_total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing (Finetuned)"):
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
    print(f"最终测试集评估 (微调后): 损失 = {final_test_loss:.4f}, 准确率 = {final_test_acc:.4f}")
    history['test_loss'] = final_test_loss
    history['test_acc'] = final_test_acc

    # 11. 可视化训练历史
    print("正在生成微调训练曲线图...")
    visualizer.plot_training_history(history, plot_name="finetune_learning_curves.png")

    print("微调流程结束。")


if __name__ == '__main__':
    run_finetuning()
    # 训练完成。总耗时: 9
    # 分
    # 36
    # 秒
    # 最佳验证准确率: 0.7990