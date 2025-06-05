import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import time
import copy
import torch.nn.functional as F
from PIL import Image


# ==============================================================================
#  函数定义部分 (这部分完全不变)
# ==============================================================================
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, device='cpu'):
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
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
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history


def plot_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_model_with_tta(model, test_loader, tta_transforms, device='cpu'):
    print("\n开始使用 TTA 在测试集上评估模型...")
    model.eval()
    total_corrects = 0
    total_images = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            image = inputs[0]
            label = labels[0].to(device)
            augmented_images = torch.stack([tf(image) for tf in tta_transforms]).to(device)
            outputs = model(augmented_images)
            mean_outputs = torch.mean(outputs, dim=0)
            _, pred = torch.max(mean_outputs, 0)
            if pred == label:
                total_corrects += 1
            total_images += 1
    tta_acc = total_corrects / total_images
    print(f'使用 TTA 的最终测试集准确率: {tta_acc:.4f} 🎯🚀')


def tta_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.tensor(labels)
    return images, labels


# ==============================================================================
#  主程序执行块
# ==============================================================================
if __name__ == '__main__':
    # 1. 设置超参数和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备是: {device} ✨")
    data_dir = 'flower_photos'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 2. 数据预处理与加载 (这部分完全不变)
    train_val_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), train_val_transforms[x])
                      for x in ['train', 'validation']}
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=None)
    image_datasets['test'] = test_dataset
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'validation']}
    test_loader_for_tta = DataLoader(image_datasets['test'], batch_size=1, shuffle=False, num_workers=4,
                                     collate_fn=tta_collate_fn)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print("数据加载完成！")
    print(
        f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['validation']}, 测试集大小: {dataset_sizes['test']}")
    print(f"类别数量: {num_classes}, 类别名称: {class_names}")

    # 3. !!! 构建模型、损失函数和优化器 (使用EfficientNet) !!!
    print("\n正在构建 EfficientNet-B0 模型...")
    # 加载预训练的 EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 冻结所有特征提取层的参数
    for param in model.parameters():
        param.requires_grad = False

    # 获取分类器的输入特征数
    # EfficientNet的分类器是一个Sequential, 我们要替换其中的Linear层(在索引1的位置)
    num_ftrs = model.classifier[1].in_features

    # 替换为我们自己的分类头
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # 将模型移动到设备
    model = model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器: 只优化我们刚刚替换的、未冻结的分类头的参数
    # 为了确保只训练分类器，可以明确地传递它的参数
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print("模型构建完成!")

    # 4. 训练和绘图
    trained_model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, device)
    print("\n正在绘制训练曲线图...")
    plot_curves(history)

    # 5. 调用TTA测试函数
    test_model_with_tta(trained_model, test_loader_for_tta, tta_transforms, device)