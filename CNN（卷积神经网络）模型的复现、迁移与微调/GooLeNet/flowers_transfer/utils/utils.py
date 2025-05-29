# flowers_transfer/utils/utils.py

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def set_seed(seed):
    """设置随机种子以确保结果可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    # 确保每次运行的CUDNN确定性行为（可能会牺牲一些性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def save_model_checkpoint(state, filepath):
    """
    更通用的模型保存函数，可以被 Trainer 内部的 save_checkpoint 调用，
    或者直接在这里实现 Trainer 的 save_checkpoint 功能。
    Trainer 中已经有了，这里可以作为备用或不同用途。
    """
    print(f"正在保存检查点到 {filepath}")
    torch.save(state, filepath)


def load_model_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """更通用的模型加载函数。"""
    if not os.path.exists(filepath):
        print(f"检查点文件未找到: {filepath}")
        return None

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"已从 {filepath} 加载模型。")
    return checkpoint


# (visualizer.py 中的绘图函数稍后添加，现在 utils.py 至少有 set_seed)

if __name__ == '__main__':
    set_seed(42)
    # 简单测试
    a = torch.randn(2, 2)
    print(a)