import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入 functional 库，用于softmax
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import time
import copy


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    # (此函数与上一版完全相同，保持不变)
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Using device: {device}')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history


# ==================== 新增的TTA测试函数 ====================
def tta_test_model(model, test_dataset):
    """
    在测试集上使用10-Crop TTA策略评估最终模型的性能
    """
    print("\nStarting evaluation on the test set with 10-Crop TTA...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 确保模型在评估模式

    # 1. 定义TTA所需的变换
    # FiveCrop会返回一个包含5个张量的元组，我们需要将它们堆叠起来
    # ToTensor和Normalize需要对每个crop都应用
    tta_transforms = transforms.Compose([
        transforms.Resize(256),
        # 产生5个裁剪图 (4个角 + 1个中心)
        transforms.TenCrop(224),
        # 将10个PIL Image的列表转换为一个[10, C, H, W]的Tensor
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # 对10个Tensor分别进行归一化
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t) for t in tensors]))
    ])

    running_corrects = 0
    total_samples = len(test_dataset)

    # 2. 遍历测试集中的每一张图片
    for i in range(total_samples):
        # 每次只取一张图片
        image, label = test_dataset[i]

        # 应用TTA变换，得到10个版本的图片
        tta_images = tta_transforms(image)  # 输出shape: [10, 3, 224, 224]

        # 将标签转为tensor，方便比较
        label_tensor = torch.tensor(label, device=device).unsqueeze(0)

        # 将10个版本的图片全部移动到GPU
        tta_images = tta_images.to(device)

        # 在无梯度的环境下进行预测
        with torch.no_grad():
            outputs = model(tta_images)  # 对10个版本进行预测, shape: [10, num_classes]

            # 3. 聚合结果
            # 使用softmax将输出转换为概率
            probs = F.softmax(outputs, dim=1)
            # 对10个版本的概率取平均值
            mean_probs = torch.mean(probs, dim=0)  # shape: [num_classes]

            # 找出概率最高的类别作为最终预测
            _, final_pred = torch.max(mean_probs, 0)

        if final_pred == label:
            running_corrects += 1

        # 打印进度
        print(f"\rEvaluating: {i + 1}/{total_samples}", end="")

    accuracy = running_corrects / total_samples

    print(f'\n\nTTA Test Set Accuracy: {running_corrects}/{total_samples} ({accuracy:.4f})')
    return accuracy


# ==============================================================


def visualize_training_curves(history, num_epochs):
    # (此函数与上一版完全相同，保持不变)
    plt.style.use("ggplot")
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(epochs_range, history['train_loss'], label="Training Loss")
    plt.plot(epochs_range, history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(epochs_range, history['train_acc'], label="Training Accuracy")
    plt.plot(epochs_range, history['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # 1. 设置超参数和数据目录
    data_dir = 'flower_photos'
    num_classes = 5
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001

    # 2. 定义数据预处理和增强
    # 注意：这里我们为TTA重新定义了变换，所以test变换只需要最基础的即可
    data_transforms = {
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
        # 测试集的变换非常简单，因为实际的复杂变换在TTA函数中定义
        'test': transforms.Compose([
            # TTA函数需要PIL Image输入，所以这里不做任何ToTensor()
        ]),
    }

    # 3. 加载数据
    print("Loading data...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'validation', 'test']}

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=False, num_workers=4)
        # 注意：我们不在主循环中使用test_loader，TTA函数会自己处理测试数据
    }
    print("Data loading complete!")

    # 4. 加载、修改和冻结模型 (与上一版相同)
    print("Loading pre-trained model AlexNet...")
    model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.classifier.parameters():
        param.requires_grad = True
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    print("Model loading and modification complete! Feature extractor is frozen.")

    # 5. 定义损失函数和优化器 (与上一版相同)
    criterion = nn.CrossEntropyLoss()
    params_to_update = [param for param in model_ft.parameters() if param.requires_grad]
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 6. 开始训练
    trained_model, history = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                                         num_epochs=num_epochs)

    # ==================== 调用新的 TTA 测试函数 ====================
    # 注意我们传递的是 test_dataset 对象，而不是 dataloader
    tta_test_model(trained_model, image_datasets['test'])
    # ==============================================================

    # 8. 保存模型
    torch.save(trained_model.state_dict(), 'alexnet_flowers_final_tta.pth')
    print("\nBest model saved to alexnet_flowers_final_tta.pth")

    # 9. 可视化结果
    print("Generating training curve plots...")
    visualize_training_curves(history, num_epochs)