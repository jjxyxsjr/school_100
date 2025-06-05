# data.py

import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import config


def split_dataset():
    """
    合并并重新划分数据集。
    首先检查新数据集目录是否已存在，如果存在则跳过划分。
    """
    if os.path.exists(config.BASE_PATH):
        print(f"数据集目录 '{config.BASE_PATH}' 已存在，跳过重新划分。")
        return

    print("开始重新划分数据集...")
    all_image_paths = []
    all_image_labels = []

    # 获取所有类别文件夹
    class_dirs = [d for d in os.listdir(config.ORIG_DATA_PATH) if os.path.isdir(os.path.join(config.ORIG_DATA_PATH, d))]

    for class_dir in class_dirs:
        # 遍历 train, test, validation 文件夹
        for split_folder in ['train', 'test', 'validation']:
            folder_path = os.path.join(config.ORIG_DATA_PATH, split_folder, class_dir)
            if os.path.exists(folder_path):
                # 使用 glob 查找所有 jpg 图片
                images = glob.glob(os.path.join(folder_path, '*.jpg'))
                for img_path in images:
                    all_image_paths.append(img_path)
                    all_image_labels.append(class_dir)

    # 第一次划分：划分为 训练集 和 (验证集+测试集)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_image_labels,
        test_size=(config.VALIDATION_RATIO + config.TEST_RATIO),
        random_state=42,
        stratify=all_image_labels  # 保持类别比例
    )

    # 第二次划分：将 temp 划分为 验证集 和 测试集
    val_ratio_in_temp = config.VALIDATION_RATIO / (config.VALIDATION_RATIO + config.TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio_in_temp),
        random_state=42,
        stratify=temp_labels  # 保持类别比例
    )

    datasets_to_create = {
        'train': (train_paths, train_labels),
        'validation': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }

    # 创建新目录并复制文件
    for split_name, (paths, labels) in datasets_to_create.items():
        split_dir = os.path.join(config.BASE_PATH, split_name)
        for i, path in enumerate(paths):
            class_dir = os.path.join(split_dir, labels[i])
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(path, class_dir)

    print("数据集划分完成！")
    print(f"新训练集大小: {len(train_paths)}")
    print(f"新验证集大小: {len(val_paths)}")
    print(f"新测试集大小: {len(test_paths)}")


def get_dataloaders():
    """
    为训练、验证和测试集创建数据加载器。
    """
    # 定义数据预处理和增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ]),
    }

    # 使用 ImageFolder 加载数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(config.BASE_PATH, x), data_transforms[x])
                      for x in ['train', 'validation', 'test']}

    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=config.BATCH_SIZE, shuffle=(x == 'train'))
                   for x in ['train', 'validation', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes

    print(f"类别: {class_names}")

    return dataloaders, dataset_sizes, class_names