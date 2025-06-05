# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def _setup_chinese_font():
    """尝试设置中文字体，以便图表能正确显示中文。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        print("注意：未找到 SimHei 字体，图表中的中文可能无法正常显示。")


def plot_stage1_curves(history):
    """
    第一阶段训练结束后调用，仅绘制第一阶段的曲线。
    """
    _setup_chinese_font()
    print("\n[可视化] 正在创建第一阶段训练曲线图...")

    s1_epochs = range(1, len(history['s1_train_loss']) + 1)

    # 创建一个包含两个子图的画布
    plt.figure(figsize=(16, 7))

    # --- 绘制左侧的 Loss 曲线图 ---
    plt.subplot(1, 2, 1)
    plt.plot(s1_epochs, history['s1_train_loss'], 'bo-', label='阶段一 训练Loss')
    plt.plot(s1_epochs, history['s1_val_loss'], 'ro-', label='阶段一 验证Loss')

    # 标记早停点
    if history.get('early_stop_epoch'):
        stop_epoch = history['early_stop_epoch']
        if stop_epoch > 0 and stop_epoch <= len(history['s1_val_loss']):
            stop_loss = history['s1_val_loss'][stop_epoch - 1]
            plt.axvline(x=stop_epoch, color='grey', linestyle='--', linewidth=2, label=f'早停点 @ Epoch {stop_epoch}')
            plt.scatter(stop_epoch, stop_loss, s=120, facecolors='none', edgecolors='red', linewidth=2, zorder=5)

    plt.title('第一阶段：训练 & 验证 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- 绘制右侧的 Accuracy 曲线图 ---
    plt.subplot(1, 2, 2)
    plt.plot(s1_epochs, np.array(history['s1_train_acc']) * 100, 'bo-', label='阶段一 训练ACC')
    plt.plot(s1_epochs, np.array(history['s1_val_acc']) * 100, 'ro-', label='阶段一 验证ACC')

    plt.title('第一阶段：训练 & 验证 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 根据要求，不再调用 plt.savefig() 和 plt.show()


def plot_final_curves(history):
    """
    第二阶段训练结束后调用，绘制包含所有阶段的完整曲线。
    """
    _setup_chinese_font()
    print("\n[可视化] 正在创建包含所有阶段的最终训练曲线图...")

    # --- 准备 x 轴数据 ---
    s1_epochs = range(1, len(history['s1_train_loss']) + 1)
    s2_epochs_range = range(len(s1_epochs), len(s1_epochs) + len(history['s2_train_loss']))

    # 创建一个包含两个子图的画布
    plt.figure(figsize=(16, 7))

    # --- 绘制左侧的 Loss 曲线图 ---
    plt.subplot(1, 2, 1)
    plt.plot(s1_epochs, history['s1_train_loss'], 'bo-', label='阶段一 训练Loss')
    plt.plot(s1_epochs, history['s1_val_loss'], 'ro-', label='阶段一 验证Loss')
    if history['s2_train_loss']:
        plt.plot(s2_epochs_range, history['s2_train_loss'], 'go-', label='阶段二 训练Loss (微调)')
        plt.plot(s2_epochs_range, history['s2_val_loss'], 'yo-', label='阶段二 验证Loss (微调)')
    if history.get('early_stop_epoch'):
        plt.axvline(x=history['early_stop_epoch'], color='grey', linestyle='--', linewidth=2, label=f'早停点')

    plt.title('完整训练过程：训练 & 验证 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- 绘制右侧的 Accuracy 曲线图 ---
    plt.subplot(1, 2, 2)
    plt.plot(s1_epochs, np.array(history['s1_train_acc']) * 100, 'bo-', label='阶段一 训练ACC')
    plt.plot(s1_epochs, np.array(history['s1_val_acc']) * 100, 'ro-', label='阶段一 验证ACC')
    if history['s2_train_acc']:
        plt.plot(s2_epochs_range, np.array(history['s2_train_acc']) * 100, 'go-', label='阶段二 训练ACC (微调)')
        plt.plot(s2_epochs_range, np.array(history['s2_val_acc']) * 100, 'yo-', label='阶段二 验证ACC (微调)')

    plt.title('完整训练过程：训练 & 验证 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 根据要求，不再调用 plt.savefig() 和 plt.show()