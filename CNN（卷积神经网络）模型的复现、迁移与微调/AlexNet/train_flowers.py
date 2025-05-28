# train_flowers.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import DataLoader
import os
import time
import copy
import json  # <-- 新增导入 for saving history
import matplotlib.pyplot as plt  # <-- 新增导入 for plotting

# 从你本地的模块导入
from models.alexnet_model import AlexNet
from data_loader import get_flowers_dataloader


def get_model(num_classes, pretrained=True, feature_extract=True):
    """
    加载或创建 AlexNet 模型，并进行迁移学习的适配。

    Args:
        num_classes (int): 新数据集的类别数。
        pretrained (bool): 是否加载 ImageNet 预训练权重。
        feature_extract (bool): 是否冻结特征提取层的权重。
                                True: 只更新分类器的权重。
                                False: 微调整个模型。
    Returns:
        torch.nn.Module: 配置好的 AlexNet 模型。
        torch.device: 使用的设备
    """
    device_to_use = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    if pretrained:
        print("加载预训练的 AlexNet 模型并进行迁移学习适配...")
        model = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)

        if feature_extract:
            print("冻结特征提取层的参数...")
            for param in model.features.parameters():
                param.requires_grad = False
            # 对于分类器，可以选择只训练最后一层，或者解冻更多层
            # 当前代码：只让最后一层可训练
            for i, param in enumerate(model.classifier.parameters()):
                if i < (len(list(model.classifier.parameters())) - 2):  # 假设最后一层是 weight 和 bias
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # 确保最后一层的 weight 和 bias 是可训练的
        else:
            print("所有参数都将参与训练 (微调模式)。")

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        print(f"分类器的最后一层已替换为输出 {num_classes} 个类别。")

    else:
        print(f"从头开始初始化 AlexNet 模型，输出 {num_classes} 个类别...")
        model = AlexNet(num_classes=num_classes)

    return model.to(device_to_use), device_to_use

# 训练模型的函数  epoch 数量可以根据需要调整
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=20,
                model_save_path="best_alexnet_flowers.pth"):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # history 字典现在存储 Python float/int 类型，方便后续 json 序列化
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()  # .item() to get Python int
                total_samples += inputs.size(0)

                if phase == 'train' and (batch_idx + 1) % 20 == 0:
                    print(f'  Batch {batch_idx + 1}/{len(dataloaders[phase])}, Loss: {loss.item():.4f}')

            epoch_loss = running_loss / total_samples
            epoch_acc = float(running_corrects) / total_samples  # Ensure float division

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_save_path)
                    print(f'新的最佳验证准确率: {best_acc:.4f}！模型已保存到 {model_save_path}')

    time_elapsed = time.time() - start_time
    print(f'\n训练完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证准确率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history


if __name__ == '__main__':
    # --- 1. 设置参数 ---
    BASE_DATA_DIR = r"D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\data\dataset"
    TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
    VAL_DIR = os.path.join(BASE_DATA_DIR, "valid")

    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20  # 初始可以设为较小值测试，例如 5
    FEATURE_EXTRACT = True
    NUM_WORKERS = 0

    MODEL_SAVE_NAME = "alexnet_flowers_best_feature_extract.pth" if FEATURE_EXTRACT else "alexnet_flowers_best_finetune.pth"
    HISTORY_SAVE_NAME = "training_history_feature_extract.json" if FEATURE_EXTRACT else "training_history_finetune.json"

    # --- 2. 加载数据 ---
    print("正在加载数据...")
    try:
        train_loader, train_dataset_obj = get_flowers_dataloader(
            data_dir=TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True, is_train=True, num_workers=NUM_WORKERS
        )
        val_loader, val_dataset_obj = get_flowers_dataloader(
            data_dir=VAL_DIR, batch_size=BATCH_SIZE, shuffle=False, is_train=False, num_workers=NUM_WORKERS
        )
        dataloaders_dict = {'train': train_loader, 'val': val_loader}
        NUM_CLASSES_FLOWERS = len(train_dataset_obj.classes)
        if NUM_CLASSES_FLOWERS == 0:
            raise ValueError("从数据加载器获取到的类别数量为0，请检查数据集和data_loader.py！")
        print(f"从数据集中检测到 {NUM_CLASSES_FLOWERS} 个类别。")
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

    # --- 3. 初始化模型 ---
    print(f"\n正在初始化模型 (特征提取模式: {FEATURE_EXTRACT})...")
    model, device_in_use = get_model(
        num_classes=NUM_CLASSES_FLOWERS, pretrained=True, feature_extract=FEATURE_EXTRACT
    )
    print(f"模型已加载到设备: {device_in_use}")

    print("\n模型中可训练的参数:")
    trainable_params_found = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            trainable_params_found = True
    if not trainable_params_found:
        print("  注意：没有参数被设置为可训练。如果这不是预期的，请检查 get_model 中的 feature_extract 逻辑。")

    # --- 4. 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    if not params_to_update:
        print("错误：没有参数被设置为可训练 (requires_grad=True)。请检查 get_model 函数中的 feature_extract 逻辑。")
        exit()
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

    # --- 5. 开始训练 ---
    print("\n开始训练...")
    trained_model, history = train_model(
        model, dataloaders_dict, criterion, optimizer, device_in_use,
        num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_NAME
    )

    print("\n训练结束!")

    # --- 6. 保存训练历史 ---
    print(f"正在保存训练历史到 {HISTORY_SAVE_NAME}...")
    try:
        with open(HISTORY_SAVE_NAME, 'w') as f:
            json.dump(history, f, indent=4)
        print("训练历史已保存。")
    except Exception as e:
        print(f"保存训练历史失败: {e}")

    # --- 7. 绘制训练和验证曲线 ---
    # 确保 history 中的值是 Python numbers, .item() 已在 train_model 中处理
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label="训练集准确率 (Train Acc)")
    plt.plot(epochs_range, history['val_acc'], label="验证集准确率 (Val Acc)")
    plt.title("训练和验证准确率 (Training and Validation Accuracy)")
    plt.xlabel("训练轮数 (Epoch)")
    plt.ylabel("准确率 (Accuracy)")
    plt.legend()
    plt.grid(True)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label="训练集损失 (Train Loss)")
    plt.plot(epochs_range, history['val_loss'], label="验证集损失 (Val Loss)")
    plt.title("训练和验证损失 (Training and Validation Loss)")
    plt.xlabel("训练轮数 (Epoch)")
    plt.ylabel("损失 (Loss)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 调整子图布局
    plt.show()