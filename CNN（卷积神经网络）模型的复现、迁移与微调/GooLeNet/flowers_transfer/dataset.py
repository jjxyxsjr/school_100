# flowers_transfer/dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np

# 导入配置
import config


class Flowers102Dataset(Dataset):
    """
    Flowers102 数据集的自定义 Dataset 类。
    它处理加载图像路径、标签和应用转换。
    """

    def __init__(self, image_dir, label_file, setid_file, split='train', transform=None):
        """
        参数:
            image_dir (str): 包含所有图像的目录。
            label_file (str): 包含图像标签的 .mat 文件路径。
            setid_file (str): 包含数据集划分 (训练/验证/测试 ID) 的 .mat 文件路径。
            split (str): 'train', 'val', 或 'test' 之一，用于指定数据集的划分。
            transform (callable, optional): 可选的转换，应用于样本。
        """
        self.image_dir = image_dir
        self.split = split.lower()
        self.transform = transform

        # 加载标签和集合 ID
        self.labels_data = scipy.io.loadmat(label_file)['labels'][0]  # 标签范围是 1-102
        setid_data = scipy.io.loadmat(setid_file)

        if self.split == 'train':
            self.image_indices = setid_data['trnid'][0]
        elif self.split == 'val' or self.split == 'valid':  # 允许 'valid' 作为别名
            self.image_indices = setid_data['valid'][0]
        elif self.split == 'test':
            self.image_indices = setid_data['tstid'][0]
        else:
            raise ValueError(f"无效的划分名称: {split}。请从 'train', 'val', 'test' 中选择。")

        # 创建一个 (image_path, label) 元组列表
        # 图像文件名格式如 'image_00001.jpg', 'image_00002.jpg', ...
        # .mat 文件中的索引是基于 1 的。
        self.samples = []
        for idx in self.image_indices:
            # 构建图像文件名: image_XXXXX.jpg，其中 XXXXX 是 5 位零填充的数字
            image_filename = f"image_{idx:05d}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)

            # .mat 文件中的标签是基于 1 的，将其转换为基于 0 的以供 PyTorch 使用
            label = int(self.labels_data[idx - 1]) - 1
            self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"未找到划分 '{self.split}' 的样本。 "
                               f"请检查路径和 .mat 文件内容: "
                               f"image_dir='{image_dir}', label_file='{label_file}', setid_file='{setid_file}'")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取给定索引处的样本 (图像和标签)。
        参数:
            idx (int): 要获取的样本的索引。
        返回:
            tuple: (image, label)，其中 image 是转换后的图像张量，
                   label 是整数标签。
        """
        image_path, label = self.samples[idx]

        try:
            image = Image.open(image_path).convert('RGB')  # 确保图像是 RGB 格式
        except FileNotFoundError:
            raise FileNotFoundError(f"在路径中未找到图像: {image_path}。"
                                    f"请验证 config.IMAGE_DIR 和图像文件名。")
        except Exception as e:
            raise RuntimeError(f"无法读取图像 {image_path}。错误: {e}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# 定义数据转换
def get_transforms(split='train'):
    """
    为给定的数据集划分返回适当的 torchvision 转换。
    参数:
        split (str): 'train', 'val', 或 'test'。
    返回:
        transforms.Compose: 转换的组合。
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 可选：更多增强
            transforms.RandomRotation(15),  # 可选：更多增强
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
        ])
    elif split == 'val' or split == 'valid' or split == 'test':
        return transforms.Compose([
            transforms.Resize(256),  # 对于 224 的中心裁剪，首先调整大小到 256
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
        ])
    else:
        raise ValueError(f"用于转换的无效划分名称: {split}")


# 创建 DataLoaders 的函数
def get_dataloaders(batch_size_train, batch_size_eval):
    """
    创建并返回训练集、验证集和测试集的 DataLoaders。
    参数:
        batch_size_train (int): 训练 DataLoader 的批量大小。
        batch_size_eval (int): 验证/测试 DataLoader 的批量大小。
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform = get_transforms('train')
    eval_transform = get_transforms('val')  # 验证集和测试集使用相同的转换

    train_dataset = Flowers102Dataset(
        image_dir=config.IMAGE_DIR,
        label_file=config.LABEL_MAT_FILE,
        setid_file=config.SETID_MAT_FILE,
        split='train',
        transform=train_transform
    )
    val_dataset = Flowers102Dataset(
        image_dir=config.IMAGE_DIR,
        label_file=config.LABEL_MAT_FILE,
        setid_file=config.SETID_MAT_FILE,
        split='val',
        transform=eval_transform
    )
    test_dataset = Flowers102Dataset(
        image_dir=config.IMAGE_DIR,
        label_file=config.LABEL_MAT_FILE,
        setid_file=config.SETID_MAT_FILE,
        split='test',
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == torch.device('cuda') else False  # 如果使用 CUDA，则固定内存
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == torch.device('cuda') else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == torch.device('cuda') else False
    )

    print(
        f"训练数据集: {len(train_dataset)} 样本, 验证数据集: {len(val_dataset)} 样本, 测试数据集: {len(test_dataset)} 样本。")
    return train_loader, val_loader, test_loader


# 示例用法 (用于直接测试此脚本)
if __name__ == '__main__':
    print("测试 Flowers102Dataset 和 DataLoaders...")

    # 测试每个划分的数据集实例化
    try:
        print("\n--- 测试训练划分 ---")
        train_dataset_test = Flowers102Dataset(
            image_dir=config.IMAGE_DIR,
            label_file=config.LABEL_MAT_FILE,
            setid_file=config.SETID_MAT_FILE,
            split='train',
            transform=get_transforms('train')
        )
        print(f"训练样本数量: {len(train_dataset_test)}")
        if len(train_dataset_test) > 0:
            img, lbl = train_dataset_test[0]
            print(f"样本图像形状: {img.shape}, 标签: {lbl.item()}")
        else:
            print("训练数据集为空。请检查 config.py 中的路径和 .mat 文件。")

        print("\n--- 测试验证划分 ---")
        val_dataset_test = Flowers102Dataset(
            image_dir=config.IMAGE_DIR,
            label_file=config.LABEL_MAT_FILE,
            setid_file=config.SETID_MAT_FILE,
            split='val',
            transform=get_transforms('val')
        )
        print(f"验证样本数量: {len(val_dataset_test)}")
        if len(val_dataset_test) > 0:
            img, lbl = val_dataset_test[0]
            print(f"样本图像形状: {img.shape}, 标签: {lbl.item()}")
        else:
            print("验证数据集为空。")

        print("\n--- 测试测试划分 ---")
        test_dataset_test = Flowers102Dataset(
            image_dir=config.IMAGE_DIR,
            label_file=config.LABEL_MAT_FILE,
            setid_file=config.SETID_MAT_FILE,
            split='test',
            transform=get_transforms('test')
        )
        print(f"测试样本数量: {len(test_dataset_test)}")
        if len(test_dataset_test) > 0:
            img, lbl = test_dataset_test[0]
            print(f"样本图像形状: {img.shape}, 标签: {lbl.item()}")
        else:
            print("测试数据集为空。")

        print("\n--- 测试 DataLoaders ---")
        # 使用虚拟批量大小进行测试
        train_loader_test, val_loader_test, test_loader_test = get_dataloaders(batch_size_train=4, batch_size_eval=4)

        print("\n从 train_loader_test 获取一个批次...")
        try:
            train_images, train_labels = next(iter(train_loader_test))
            print(f"训练批次图像形状: {train_images.shape}, 训练批次标签形状: {train_labels.shape}")
        except RuntimeError as e:
            if "Tried to open" in str(e) or "No samples found" in str(e) or "Image not found" in str(e):
                print(f"创建 train_loader 时出错: {e}")
                print("这可能意味着图像路径不正确或图像丢失。")
                print(f"请验证 config.IMAGE_DIR: {config.IMAGE_DIR}")
                print(f"并确保类似 '{os.path.join(config.IMAGE_DIR, 'image_00001.jpg')}' 的文件存在。")
            else:
                raise e

        print("\n从 val_loader_test 获取一个批次...")
        try:
            val_images, val_labels = next(iter(val_loader_test))
            print(f"验证批次图像形状: {val_images.shape}, 验证批次标签形状: {val_labels.shape}")
        except RuntimeError as e:
            if "Tried to open" in str(e) or "No samples found" in str(e) or "Image not found" in str(e):
                print(f"创建 val_loader 时出错: {e}")
            else:
                raise e

    except FileNotFoundError as e:
        print(f"\n错误: 未找到 .mat 文件或图像目录。")
        print(f"详细信息: {e}")
        print("请确保 'config.py' 中的路径正确无误:")
        print(f"  config.IMAGE_DIR = '{config.IMAGE_DIR}'")
        print(f"  config.LABEL_MAT_FILE = '{config.LABEL_MAT_FILE}'")
        print(f"  config.SETID_MAT_FILE = '{config.SETID_MAT_FILE}'")
    except Exception as e:
        print(f"\n测试过程中发生意外错误: {e}")
        import traceback

        traceback.print_exc()