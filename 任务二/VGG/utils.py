# utils.py

import matplotlib.pyplot as plt


def plot_training_results(history, stage_num, save_path):
    """
    可视化训练和验证的 Loss 和 Accuracy 曲线。

    Args:
        history (dict): 包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 的字典。
        stage_num (int): 训练阶段编号 (1 或 2)。
        save_path (str): 图像保存路径。
    """
    # 设置 matplotlib 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'阶段 {stage_num}: 训练过程可视化', fontsize=16)

    epochs_range = range(len(history['train_loss']))

    # 绘制 Loss 曲线
    ax1.plot(epochs_range, history['train_loss'], label='训练 Loss')
    ax1.plot(epochs_range, history['val_loss'], label='验证 Loss')
    ax1.set_title('训练 & 验证 Loss 曲线图')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # 绘制 Accuracy 曲线
    ax2.plot(epochs_range, history['train_acc'], label='训练 Accuracy')
    ax2.plot(epochs_range, history['val_acc'], label='验证 Accuracy')
    ax2.set_title('训练 & 验证 Accuracy 曲线图')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"训练可视化图表已保存至: {save_path}")
    plt.close()  # 关闭画布，防止在Jupyter等环境中重复显示