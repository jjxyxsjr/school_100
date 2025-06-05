# src/visualizer.py
# 管理训练历史的可视化和保存

import matplotlib.pyplot as plt
import json

class Visualizer:
    def __init__(self, config):
        """初始化可视化器，设置配置"""
        self.config = config

    def plot_training_history(self, history_dict, initial_epochs_count):
        """绘制训练和验证的准确率及损失曲线"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False

        acc = history_dict.get('accuracy', [])
        val_acc = history_dict.get('val_accuracy', [])
        loss = history_dict.get('loss', [])
        val_loss = history_dict.get('val_loss', [])

        # 转换张量为标量
        acc = [a.item() if isinstance(a, torch.Tensor) else a for a in acc]
        val_acc = [a.item() if isinstance(a, torch.Tensor) else a for a in val_acc]
        loss_vals = [l.item() if isinstance(a, torch.Tensor) else l for l in loss]
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

    def save_history(self, history_dict):
        """保存训练历史到 JSON 文件"""
        with open(self.config.HISTORY_SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=4, ensure_ascii=False)
