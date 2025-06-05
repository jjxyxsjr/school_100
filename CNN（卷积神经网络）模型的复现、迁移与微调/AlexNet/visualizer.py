"""
可视化模块：绘制训练和验证的损失及准确率曲线。
- 生成两张图：损失曲线和准确率曲线。
- 帮助分析模型的训练过程和性能。
"""
# visualizer.py
import matplotlib.pyplot as plt
# --- 添加下面这两行代码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# -------------------------

class TrainingVisualizer:
    @staticmethod
    def visualize_training_curves(history, num_epochs):
        plt.style.use("ggplot")
        epochs_range = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.title("训练和验证损失")
        plt.plot(epochs_range, history['train_loss'], label="训练损失")
        plt.plot(epochs_range, history['val_loss'], label="验证损失")
        plt.xlabel("Epochs")
        plt.ylabel("损失")
        plt.legend(loc="upper right")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.title("训练和验证准确率")
        plt.plot(epochs_range, history['train_acc'], label="训练准确率")
        plt.plot(epochs_range, history['val_acc'], label="验证准确率")
        plt.xlabel("Epochs")
        plt.ylabel("准确率")
        plt.legend(loc="lower right")
        plt.show()