"""
DataLoaderFactory class for loading and preprocessing the flower photos dataset.
Handles data transformations, dataset creation, and DataLoader setup for training, validation, and TTA testing.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
class DataLoaderFactory:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def create_dataloaders(self):
        # 数据预处理变换
        train_val_transforms = {
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
        }

        # TTA 变换
        tta_transforms = [
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
            ]),
        ]

        # 加载数据集
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), train_val_transforms[x])
            for x in ['train', 'validation']
        }
        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=None)
        image_datasets['test'] = test_dataset

        # 创建 DataLoader
        dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'validation']
        }
        dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=1, shuffle=False, num_workers=4,
                                         collate_fn=self._tta_collate_fn)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
        class_names = image_datasets['train'].classes

        return dataloaders, dataset_sizes, class_names, tta_transforms

    @staticmethod
    def _tta_collate_fn(batch):
        """Custom collate function for TTA testing, processes single images and labels."""
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        labels = torch.tensor(labels)
        return images, labels