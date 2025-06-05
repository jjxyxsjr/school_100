"""
Visualizer class for plotting training and validation accuracy and loss curves.
Provides visualization of model performance over epochs.
"""

import matplotlib.pyplot as plt

class Visualizer:
    def plot_curves(self, history):
        """Plots accuracy and loss curves for training and validation."""
        print("\n正在绘制训练曲线图...")
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