"""
数据处理模块，负责加载和预处理图像数据集，创建DataLoader，并定义TTA变换。
提供数据集加载、变换配置和自定义collate_fn功能，用于支持训练、验证和TTA测试。
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class DataHandler:
    """负责数据加载和预处理"""
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_val_transforms = {
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
            ])
        }
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]

    def load_data(self):
        """加载数据集并创建DataLoader"""
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.train_val_transforms[x])
            for x in ['train', 'validation']
        }
        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=None)
        image_datasets['test'] = test_dataset

        dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'validation']
        }
        test_loader = DataLoader(image_datasets['test'], batch_size=1, shuffle=False,
                                 num_workers=4, collate_fn=self.tta_collate_fn)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
        class_names = image_datasets['train'].classes
        return image_datasets, dataloaders, test_loader, dataset_sizes, class_names

    @staticmethod
    def tta_collate_fn(batch):
        """自定义collate_fn，用于TTA测试"""
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        return images, labels