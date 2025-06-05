"""
数据加载模块：处理训练、验证和测试的数据加载与预处理。
- 定义训练、验证和测试的数据变换。
- 使用 ImageFolder 加载数据集，并为训练和验证创建 DataLoader。
"""
# data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class FlowerDataLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_transforms = {
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
            'test': transforms.Compose([
                # 测试时变换在 TTA 中处理，此处不应用复杂变换
            ])
        }

    # 数据变换策略 ：
    # 训练：随机裁剪、翻转 + 归一化
    # 验证：中心裁剪 + 归一化
    # 测试：未定义完整变换（TTA
    # 中处理）
    # 使用字典统一管理不同阶段的
    # transform。
    # 批量大小和
    # num_workers
    # 固定为
    # 32
    # 和
    # 4
    def load_data(self):
        print("加载数据...")
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
            for x in ['train', 'validation', 'test']
        }
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=4),
            'validation': DataLoader(image_datasets['validation'], batch_size=self.batch_size, shuffle=False, num_workers=4)
            # 测试集的 DataLoader 在 TTA 中单独处理
        }
        print("数据加载完成！")
        return image_datasets, dataloaders