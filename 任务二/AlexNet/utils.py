# utils.py

import matplotlib.pyplot as plt
import torch


def plot_curves(history, stage_name):
    """
    绘制训练和验证的损失及准确率曲线图。
    Args:
        history (dict): 包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 的字典。
        stage_name (str): 阶段名称，如 "stage1" 或 "stage2"。
    """
    # --- 支持中文显示 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    epochs = range(1, len(train_loss) + 1)

    # 创建一个画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 绘制 Loss 曲线 ---
    ax1.plot(epochs, train_loss, 'bo-', label='训练 Loss')
    ax1.plot(epochs, val_loss, 'ro-', label='验证 Loss')
    ax1.set_title(f'阶段 {stage_name[-1]}: 训练 & 验证 Loss 曲线图', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # --- 绘制 Accuracy 曲线 ---
    ax2.plot(epochs, train_acc, 'bo-', label='训练 Accuracy')
    ax2.plot(epochs, val_acc, 'ro-', label='验证 Accuracy')
    ax2.set_title(f'阶段 {stage_name[-1]}: 训练 & 验证 Accuracy 曲线图', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    # 保存图片
    save_path = f"training_curves_{stage_name}.png"
    plt.savefig(save_path)
    print(f"训练曲线图已保存至: {save_path}")
    plt.show()


def test_model(model, test_loader, device):
    """
    在测试集上评估模型性能。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy