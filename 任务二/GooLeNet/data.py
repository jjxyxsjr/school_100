# -*- coding: utf-8 -*-
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from 任务二.GooLeNet import config

# --- 图像预处理与增强 ---
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_transform_stage1 = val_test_transform
train_transform_stage2 = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders():
    """
    加载总数据集，并按80/10/10划分为训练、验证、测试集。
    然后将训练集按70/30划分为两个部分。
    """
    full_dataset = datasets.ImageFolder(config.DATA_DIR)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    test_split_point = int(np.floor(config.TEST_SPLIT * dataset_size))
    validation_split_point = int(np.floor(config.VALIDATION_SPLIT * dataset_size)) + test_split_point

    test_indices = indices[:test_split_point]
    val_indices = indices[test_split_point:validation_split_point]
    train_indices = indices[validation_split_point:]

    train_size = len(train_indices)
    split_point_train = int(np.floor(config.TRAIN_SPLIT_RATIO * train_size))
    train_part1_indices = train_indices[:split_point_train]
    train_part2_indices = train_indices[split_point_train:]

    dataset_s1 = datasets.ImageFolder(config.DATA_DIR, transform=train_transform_stage1)
    dataset_s2 = datasets.ImageFolder(config.DATA_DIR, transform=train_transform_stage2)
    dataset_val_test = datasets.ImageFolder(config.DATA_DIR, transform=val_test_transform)

    train_subset1 = Subset(dataset_s1, train_part1_indices)
    train_subset2 = Subset(dataset_s2, train_part2_indices)
    val_subset = Subset(dataset_val_test, val_indices)
    test_subset = Subset(dataset_val_test, test_indices)

    train_loader1 = DataLoader(train_subset1, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
                               pin_memory=True)
    train_loader2 = DataLoader(train_subset2, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
                               pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"总数据集大小: {dataset_size}")
    print(
        f"新训练集大小: {len(train_indices)} (Stage 1: {len(train_part1_indices)}, Stage 2: {len(train_part2_indices)})")
    print(f"新验证集大小: {len(val_indices)}")
    print(f"新测试集大小: {len(test_indices)}")

    return train_loader1, train_loader2, val_loader, test_loader