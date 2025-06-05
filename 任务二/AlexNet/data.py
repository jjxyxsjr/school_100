# data.py

import os
import shutil
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import config


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*'):
                if is_valid_image(img_path):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


def split_dataset():
    origin_path = Path(config.ORIGIN_DATA_PATH)
    target_path = Path(config.DATA_ROOT)

    if target_path.exists():
        print(f"'{target_path}' 目录已存在，跳过数据集划分。")
        return

    print(f"'{target_path}' 目录不存在，开始创建和划分数据集...")
    all_files = []
    class_names = [d.name for d in origin_path.glob('train/*') if d.is_dir()]

    for split in ['train', 'validation', 'test']:
        split_dir = origin_path / split
        if not split_dir.is_dir(): continue
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.is_dir(): continue
            for img_path in class_dir.glob('*'):
                if is_valid_image(img_path):
                    all_files.append((img_path, class_name))

    random.shuffle(all_files)
    total_size = len(all_files)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)

    for split in ['train', 'validation', 'test']:
        for class_name in class_names:
            (target_path / split / class_name).mkdir(parents=True, exist_ok=True)

    splits = {
        'train': all_files[:train_size],
        'validation': all_files[train_size:train_size + val_size],
        'test': all_files[train_size + val_size:]
    }

    for split_name, files in splits.items():
        for img_path, class_name in files:
            shutil.copy(img_path, target_path / split_name / class_name / img_path.name)

    print("数据集划分完成！")


# 💡 修改: 移除了 use_augmentation 参数和相关逻辑
def get_dataloaders():
    """
    获取训练、验证和测试的数据加载器。
    所有数据集使用相同的图像变换。
    """
    image_size = 224

    # 统一使用基础变换
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = Path(config.DATA_ROOT)
    # 所有数据集都使用同一种 transform
    train_dataset = ImageDataset(root_dir=data_dir / 'train', transform=data_transform)
    val_dataset = ImageDataset(root_dir=data_dir / 'validation', transform=data_transform)
    test_dataset = ImageDataset(root_dir=data_dir / 'test', transform=data_transform)

    num_workers = 0  # 之前确认过设为0可以稳定运行
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader