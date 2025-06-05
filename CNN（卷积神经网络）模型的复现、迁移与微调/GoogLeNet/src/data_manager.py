# src/data_manager.py
# 管理数据加载、变换和 DataLoader 创建

import torch
from torchvision import datasets, transforms
from torchvision.transforms import TrivialAugmentWide
from torch.utils.data import DataLoader
import os

class DataManager:
    def __init__(self, config):
        """初始化数据管理器，设置配置和变换"""
        self.config = config
        self.data_transforms = self._get_transforms()  # 获取数据变换
        self.image_datasets = {}  # 存储数据集
        self.dataloaders = {}  # 存储 DataLoader
        self.dataset_sizes = {}  # 存储数据集大小

    def _get_transforms(self):
        """定义训练、验证和测试的数据变换"""
        return {
            'train': transforms.Compose([
                TrivialAugmentWide(),  # 高级数据增强
                transforms.Resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 归一化
            ]),
            'validation': transforms.Compose([
                transforms.Resize(self.config.IMG_HEIGHT + 1),
                transforms.CenterCrop(self.config.IMG_WIDTH),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def load_data(self):
        """加载数据集并创建 DataLoader"""
        # 加载训练和验证数据集
        self.image_datasets = {
            'train': datasets.ImageFolder(self.config.TRAIN_DIR, self.data_transforms['train']),
            'validation': datasets.ImageFolder(self.config.VALID_DIR, self.data_transforms['validation'])
        }
        self.dataloaders = {
            'train': DataLoader(self.image_datasets['train'], batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=4),
            'validation': DataLoader(self.image_datasets['validation'], batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4)
        }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'validation']}

        # 加载测试集（如果存在）
        if os.path.exists(self.config.TEST_DIR) and len(os.listdir(self.config.TEST_DIR)) > 0:
            self.image_datasets['test'] = datasets.ImageFolder(self.config.TEST_DIR, self.data_transforms['test'])
            self.dataloaders['test'] = DataLoader(self.image_datasets['test'], batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4)
            self.dataset_sizes['test'] = len(self.image_datasets['test'])
            print(f"测试集加载成功，包含 {self.dataset_sizes['test']} 个样本。")
