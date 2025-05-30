# finetune_project/train_finetune.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import json
import copy


def run_finetune():
    # --- 1. 配置 ---
    # 数据集路径 (请确保这个路径是正确的)
    dataset_path = r'D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\data\flowers102_structured'

    # !! 加载之前训练好的最佳模型 !!
    # 使用相对路径，假设模型文件在当前目录的上一级
    source_model_path = '../best_model_enhanced.pth'

    # 新的输出文件，将保存在当前脚本所在的目录
    json_output_file = 'finetune_metrics.json'
    best_model_path = 'finetune_best_model.pth'

    # 类别数
    num_classes = 102

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 微调超参数
    batch_size = 64
    finetune_epochs = 50
    patience = 5
    lr_features = 1e-5  # 卷积层的学习率
    lr_classifier = 1e-4  # 分类头的学习率

    # --- 2. 模型定义与微调设置 ---
    print("\n--- 正在设置微调模型 ---")

    # 首先，创建一个与保存时结构相同的“空”模型
    # 这只是为了有一个正确的“骨架”来加载权重
    model = models.alexnet(weights=None)  # 不使用预训练权重，因为我们要加载自己的
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # 加载我们之前训练好的模型的状态字典
    try:
        model.load_state_dict(torch.load(source_model_path, map_location=device))
        print(f"成功从 {source_model_path} 加载模型权重。")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {source_model_path}。请检查文件路径。")
        return

    # --- 这里是微调的核心设置 ---!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 1. 解冻所有层，使它们都可以在训练中被更新
    for param in model.parameters():
        param.requires_grad = True
    print("模型所有层已解冻 (features + classifier)。")
    # 2. 没有任何层被“替换”，我们是在已训练好的分类头基础上继续训练。
    # --------------------------

    model = model.to(device)

    # --- 3. 数据加载与预处理 ---
    print("\n--- 正在加载数据 ---")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x]) for x in
                      ['train', 'valid']}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    # print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['valid']}")

    # --- 4. 优化器与损失函数 ---
    # 设置差分学习率
    params_to_update = [
        {'params': model.features.parameters(), 'lr': lr_features},
        {'params': model.classifier.parameters(), 'lr': lr_classifier}
    ]
    optimizer = optim.Adam(params_to_update, lr=lr_classifier)  # 默认lr可以设为分类头的lr
    criterion = nn.CrossEntropyLoss()

    # --- 5. 训练循环 ---
    print("\n--- 开始微调训练 ---")
    training_history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(finetune_epochs):
        epoch_start_time = time.time()
        print(f"\n周期 {epoch}/{finetune_epochs - 1}")
        print("-" * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}")

            if phase == 'train':
                training_history['train_loss'].append(epoch_loss)
                training_history['train_acc'].append(epoch_acc.item())
            else:
                training_history['valid_loss'].append(epoch_loss)
                training_history['valid_acc'].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"新最佳模型已保存至 {best_model_path}，验证准确率: {best_val_acc:.4f}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        with open(json_output_file, 'w') as f:
            json.dump(training_history, f, indent=4)

        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"周期 {epoch} 在 {epoch_time_elapsed // 60:.0f}分 {epoch_time_elapsed % 60:.2f}秒 内完成")

        if epochs_no_improve >= patience:
            print(f"\n早停触发！连续 {patience} 个周期验证准确率未提升。")
            break

    print(f"\n微调完成。最佳验证准确率: {best_val_acc:.4f}")
    print(f"训练指标已保存至 {json_output_file}")
    print(f"最佳模型已保存至 {best_model_path}")


if __name__ == '__main__':
    run_finetune()