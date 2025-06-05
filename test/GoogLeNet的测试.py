import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
# 导入新增的模块
from torchvision.transforms import TrivialAugmentWide
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
import copy
import csv
import multiprocessing

# --- 全局常量定义 ---
IMG_WIDTH, IMG_HEIGHT = 299, 299
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS_INITIAL = 15
EPOCHS_FINE_TUNE = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

# --- 优化后的超参数 ---
OPTIMIZED_DROPOUT_RATE = 0.6
OPTIMIZED_WEIGHT_DECAY = 1e-4

BASE_DIR = './flower_photos'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# 更新文件名以反映新的优化
MODEL_SAVE_PATH = '../CNN（卷积神经网络）模型的复现、迁移与微调/GoogLeNet/best_flower_classifier_final_optimized.pth'
HISTORY_SAVE_PATH = '../CNN（卷积神经网络）模型的复现、迁移与微调/GoogLeNet/training_history_final_optimized.json'
CSV_LOG_PATH = '../CNN（卷积神经网络）模型的复现、迁移与微调/GoogLeNet/training_log_final_optimized.csv'


# --- 函数定义 ---
# train_model 和 plot_training_history 函数保持不变，这里为了简洁省略其代码，
# 您可以从上一版代码中直接使用。
# 为确保完整性，我还是将它们包含进来。

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, dataset_sizes,
                phase_name="训练", current_best_acc=0.0, patience=5, csv_log_path=CSV_LOG_PATH,
                model_save_path=MODEL_SAVE_PATH):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = current_best_acc

    write_header = not os.path.exists(csv_log_path) or os.path.getsize(csv_log_path) == 0
    with open(csv_log_path, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(['epoch', 'phase', 'loss', 'accuracy', 'timestamp'])

        epochs_no_improve = 0
        stopped_epoch = num_epochs - 1

        for epoch in range(num_epochs):
            print(f'\n轮次 {epoch + 1}/{num_epochs} ({phase_name})')
            print('-' * 10)

            for phase in ['train', 'validation']:
                phase_cn = "训练" if phase == 'train' else "验证"
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
                        if phase == 'train' and model.aux_logits and hasattr(model,
                                                                             'AuxLogits') and model.AuxLogits is not None:
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                print(f'{phase_cn}损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')
                csv_writer.writerow([epoch, phase, f"{epoch_loss:.4f}", f"{epoch_acc:.4f}", current_time_str])
                csv_file.flush()

                if phase == 'validation':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), model_save_path)
                        print(f"最佳验证准确率提升至 {best_acc:.4f}。模型已保存至 {model_save_path}")
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                else:
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())

            if epochs_no_improve >= patience:
                print(f'提前终止于第 {epoch + 1} 轮。')
                stopped_epoch = epoch
                model.load_state_dict(best_model_wts)
                return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history, best_acc, stopped_epoch

    time_elapsed = time.time() - since
    print(f'{phase_name}阶段在 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒 内完成')
    print(f'最佳验证准确率: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history, best_acc, stopped_epoch


def plot_training_history(history_dict_to_plot, initial_epochs_count):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    acc = history_dict_to_plot.get('accuracy', [])
    val_acc = history_dict_to_plot.get('val_accuracy', [])
    loss = history_dict_to_plot.get('loss', [])
    val_loss = history_dict_to_plot.get('val_loss', [])

    acc = [a.item() if isinstance(a, torch.Tensor) else a for a in acc]
    val_acc = [a.item() if isinstance(a, torch.Tensor) else a for a in val_acc]
    loss_vals = [l.item() if isinstance(l, torch.Tensor) else l for l in loss]
    val_loss_vals = [l.item() if isinstance(l, torch.Tensor) else l for l in val_loss]

    total_epochs = len(acc)
    if total_epochs == 0:
        print("历史记录中无数据可供绘制。")
        return

    epochs_range = range(total_epochs)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='训练准确率')
    plt.plot(epochs_range, val_acc, label='验证准确率')
    if initial_epochs_count > 0 and initial_epochs_count < total_epochs:
        plt.axvline(x=initial_epochs_count - 1, color='grey', linestyle='--', label='开始微调')
    plt.legend(loc='lower right')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次 (Epochs)')
    plt.ylabel('准确率')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_vals, label='训练损失')
    plt.plot(epochs_range, val_loss_vals, label='验证损失')
    if initial_epochs_count > 0 and initial_epochs_count < total_epochs:
        plt.axvline(x=initial_epochs_count - 1, color='grey', linestyle='--', label='开始微调')
    plt.legend(loc='upper right')
    plt.title('训练和验证损失')
    plt.xlabel('轮次 (Epochs)')
    plt.ylabel('损失')
    plt.grid(True)

    plt.suptitle("模型训练历史 (最终优化版)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# 主执行逻辑
if __name__ == '__main__':
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # --- 数据准备与增强 ---
    print("正在准备数据转换和加载器 (使用 TrivialAugment)...")
    data_transforms = {
        # 新增改动 1: 使用更高级的 TrivialAugmentWide 进行训练数据增强
        'train': transforms.Compose([
            TrivialAugmentWide(),
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # TrivialAugment后需要确保尺寸统一
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(IMG_HEIGHT + 1),
            transforms.CenterCrop(IMG_WIDTH),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # TTA手动处理翻转，这里直接resize
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'validation': datasets.ImageFolder(VALID_DIR, data_transforms['validation'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

    if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        image_datasets['test'] = datasets.ImageFolder(TEST_DIR, data_transforms['test'])
        dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        dataset_sizes['test'] = len(image_datasets['test'])
        print(f"测试集加载成功，包含 {dataset_sizes['test']} 个样本。")

    # --- 构建模型 (与上一版相同) ---
    print("\n正在构建模型...")
    model_ft = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs_main = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs_main, 1024), nn.ReLU(),
        nn.Dropout(p=OPTIMIZED_DROPOUT_RATE), nn.Linear(1024, NUM_CLASSES)
    )
    if hasattr(model_ft, 'AuxLogits') and model_ft.AuxLogits is not None:
        num_ftrs_aux = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Sequential(
            nn.Linear(num_ftrs_aux, 1024), nn.ReLU(),
            nn.Dropout(p=OPTIMIZED_DROPOUT_RATE), nn.Linear(1024, NUM_CLASSES)
        )
    model_ft = model_ft.to(device)

    # --- 初始训练 (与上一版相同) ---
    print("\n为特征提取阶段配置优化器...")
    params_to_update_initial = [param for param in model_ft.parameters() if param.requires_grad]
    optimizer_initial = optim.Adam(params_to_update_initial, lr=INITIAL_LR, weight_decay=OPTIMIZED_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    print("\n开始初始训练 (特征提取)...")
    model_ft, train_acc_initial, val_acc_initial, train_loss_initial, val_loss_initial, best_val_acc_initial, stopped_epoch_initial = train_model(
        model_ft, dataloaders, criterion, optimizer_initial, EPOCHS_INITIAL, device, dataset_sizes,
        phase_name="初始训练", patience=5
    )
    actual_initial_epochs = stopped_epoch_initial + 1

    # --- 微调 (与上一版相同) ---
    print("\n准备进行微调...")
    fine_tune_after_layer_name = 'Mixed_7a'
    found_fine_tune_layer = False
    for name, child in model_ft.named_children():
        if fine_tune_after_layer_name in name:
            found_fine_tune_layer = True
        if found_fine_tune_layer:
            for param in child.parameters():
                param.requires_grad = True
    for param in model_ft.fc.parameters():
        param.requires_grad = True
    if hasattr(model_ft, 'AuxLogits') and model_ft.AuxLogits is not None:
        for param in model_ft.AuxLogits.fc.parameters():
            param.requires_grad = True
    params_to_update_ft = filter(lambda p: p.requires_grad, model_ft.parameters())
    optimizer_fine_tune = optim.Adam(params_to_update_ft, lr=FINE_TUNE_LR, weight_decay=OPTIMIZED_WEIGHT_DECAY)
    print("\n开始微调...")
    model_ft, train_acc_ft, val_acc_ft, train_loss_ft, val_loss_ft, best_val_acc_ft, stopped_epoch_ft = train_model(
        model_ft, dataloaders, criterion, optimizer_fine_tune, EPOCHS_FINE_TUNE, device, dataset_sizes,
        phase_name="微调", current_best_acc=best_val_acc_initial, patience=7
    )

    # --- 结果处理与保存 ---
    print("\n正在保存训练历史...")
    history_initial = {'accuracy': train_acc_initial, 'val_accuracy': val_acc_initial, 'loss': train_loss_initial,
                       'val_loss': val_loss_initial}
    history_fine_tune = {'accuracy': train_acc_ft, 'val_accuracy': val_acc_ft, 'loss': train_loss_ft,
                         'val_loss': val_loss_ft}
    combined_history = {key: history_initial[key] + history_fine_tune.get(key, []) for key in history_initial.keys()}
    with open(HISTORY_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(combined_history, f, indent=4, ensure_ascii=False)
    print(f"训练历史已保存至: {HISTORY_SAVE_PATH}")

    # --- 最终评估 (使用 TTA) ---
    print("\n正在加载最佳模型并使用TTA进行最终评估...")
    model_ft.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model_ft.eval()

    if 'test' in dataloaders:
        test_loss = 0.0
        test_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # --- 新增改动 2: TTA 逻辑 ---
                # 1. 原始图像预测
                outputs_original = model_ft(inputs)
                # 损失基于原始图像计算
                loss_val = criterion(outputs_original, labels)
                probs_original = torch.softmax(outputs_original, dim=1)

                # 2. 水平翻转图像预测
                flipped_inputs = F.hflip(inputs)
                outputs_flipped = model_ft(flipped_inputs)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # 3. 平均两次预测的概率
                avg_probs = (probs_original + probs_flipped) / 2
                _, preds = torch.max(avg_probs, 1)

                # 累加损失和正确数
                test_loss += loss_val.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)

        final_test_loss = test_loss / dataset_sizes['test']
        final_test_acc = test_corrects.double() / dataset_sizes['test']
        print(f"\n最终测试集表现 (TTA):")
        print(f"测试损失: {final_test_loss:.4f}")
        print(f"测试准确率: {final_test_acc * 100:.2f}%")

    # --- 可视化 ---
    print("\n正在绘制训练历史图...")
    plot_training_history(combined_history, actual_initial_epochs)

    print("\n最终优化版 PyTorch 脚本执行完毕。")