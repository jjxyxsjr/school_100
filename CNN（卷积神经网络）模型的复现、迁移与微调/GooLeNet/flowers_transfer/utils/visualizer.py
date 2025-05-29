# flowers_transfer/utils/visualizer.py

import matplotlib.pyplot as plt
import os
import config # 用于获取 PLOT_DIR

# 尝试设置支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
    plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"无法设置中文字体 'SimHei'，图表中的中文可能无法正确显示。错误: {e}")
    print("请确保你的系统中安装了 SimHei 字体，或者更换为其他可用的中文字体。")

def plot_training_history(history, plot_name="training_curves.png"):
    """
    绘制并保存训练过程中的损失和准确率曲线。

    参数:
        history (dict): 包含 'train_loss', 'train_acc', 'val_loss', 'val_acc' 列表的字典。
        plot_name (str): 保存绘图的文件名。
    """
    if not all(key in history for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']):
        print("错误: history 字典缺少必要的键 ('train_loss', 'train_acc', 'val_loss', 'val_acc')")
        return

    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='训练损失 (Train Loss)')
    plt.plot(epochs_range, history['val_loss'], label='验证损失 (Validation Loss)')
    plt.title('损失曲线 (Loss Curves)')
    plt.xlabel('周期 (Epochs)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='训练准确率 (Train Accuracy)')
    plt.plot(epochs_range, history['val_acc'], label='验证准确率 (Validation Accuracy)')
    plt.title('准确率曲线 (Accuracy Curves)')
    plt.xlabel('周期 (Epochs)')
    plt.ylabel('准确率 (Accuracy)')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'训练历史: {plot_name.split(".")[0]}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为 suptitle 调整布局

    # 保存图像
    save_path = os.path.join(config.PLOT_DIR, plot_name)
    os.makedirs(config.PLOT_DIR, exist_ok=True) # 确保目录存在
    plt.savefig(save_path)
    print(f"训练曲线图已保存到: {save_path}")
    plt.close() # 关闭图像，防止在某些环境下直接显示

if __name__ == '__main__':
    # 简单测试 visualizer
    print("Visualizer.py: 定义了绘图函数。")
    dummy_history = {
        'train_loss': [0.5, 0.4, 0.3],
        'train_acc': [0.7, 0.8, 0.9],
        'val_loss': [0.45, 0.35, 0.25],
        'val_acc': [0.75, 0.85, 0.92],
        'lr': [0.001, 0.001, 0.0001]
    }
    plot_training_history(dummy_history, plot_name="dummy_test_plot_chinese_font.png")