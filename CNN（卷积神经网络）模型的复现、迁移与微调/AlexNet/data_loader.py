# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os  # Added for path joining

# AlexNet 期望的输入尺寸
INPUT_SIZE = 224

# ImageNet 图像的均值和标准差，用于归一化
# 这些值是广泛使用的标准值
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_flowers_dataloader(data_dir, batch_size, shuffle=True, num_workers=4, is_train=True):
    """
    为Flowers数据集创建 DataLoader。

    Args:
        data_dir (str): 数据集所在的根目录路径 (例如 'path/to/flowers_dataset/train' 或 'path/to/flowers_dataset/valid')。
        batch_size (int): 每个批次的样本数。
        shuffle (bool): 是否在每个epoch开始时打乱数据。通常训练时为True，验证/测试时为False。
        num_workers (int): 用于数据加载的子进程数量。可以根据CPU核心数调整。
                           在Windows上，如果 num_workers > 0 遇到问题，可以尝试设为0。
        is_train (bool): 指示是否为训练集加载器，用于应用不同的transform。

    Returns:
        torch.utils.data.DataLoader: 配置好的数据加载器。
        torchvision.datasets.ImageFolder: 数据集对象，可以用来获取类别信息。
    """
    if is_train:
        # 训练集的数据增强和预处理
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        # 验证/测试集的预处理 (不需要数据增强)
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    try:
        image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    except FileNotFoundError:
        print(f"错误：找不到数据集路径 {data_dir}。请检查路径是否正确。")
        raise
    except Exception as e:
        print(f"加载数据集 {data_dir} 时发生错误: {e}")
        raise

    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True if torch.cuda.is_available() else False)

    print(f"成功加载 {'训练' if is_train else '验证/测试'} 数据集来源: {data_dir}")
    print(f"  样本数量: {len(image_dataset)}")
    print(f"  类别数量: {len(image_dataset.classes)}")
    # print(f"  类别到索引的映射: {image_dataset.class_to_idx}") # Uncomment to see mapping

    return dataloader, image_dataset


if __name__ == '__main__':
    # --- 用于测试 data_loader.py 的示例代码 ---

    # 基础路径，指向你项目中的 data 文件夹的上层目录
    # D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet
    # 我们需要的是 D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\data\dataset

    # 使用你的实际路径结构
    # 注意：在Python字符串中，反斜杠 \ 是转义字符，所以路径可以用 / 或者 \\
    base_data_dir = r"D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\data\dataset"

    train_data_path = os.path.join(base_data_dir, "train")
    val_data_path = os.path.join(base_data_dir, "valid")  # Your validation folder is named "valid"
    test_data_path = os.path.join(base_data_dir, "test")  # You also have a "test" folder

    BATCH_SIZE_TEST = 4
    NUM_WORKERS_TEST = 0  # 设为0以避免Windows上多进程的潜在问题，尤其是在简单测试时

    print(f"预期训练数据路径: {train_data_path}")
    print(f"预期验证数据路径: {val_data_path}")
    print(f"预期测试数据路径: {test_data_path}")

    print("\n正在测试训练数据加载器 (train_loader)...")
    if os.path.exists(train_data_path):
        try:
            train_loader, train_dataset = get_flowers_dataloader(
                data_dir=train_data_path,
                batch_size=BATCH_SIZE_TEST,
                shuffle=True,
                is_train=True,
                num_workers=NUM_WORKERS_TEST
            )
            images, labels = next(iter(train_loader))
            print(f"  训练集批次图像形状: {images.shape}")
            print(f"  训练集批次标签形状: {labels.shape}")
            print(f"  训练集检测到的类别名称示例 (前5个): {train_dataset.classes[:5]}")
            detected_num_classes = len(train_dataset.classes)
            print(f"  训练集检测到的类别数量: {detected_num_classes}")
            if detected_num_classes != 102:  # Assuming 102 classes from your folder names
                print(f"  警告: 检测到的类别数量 ({detected_num_classes}) 与预期的102不符。请检查数据集。")

        except Exception as e:
            print(f"  测试训练数据加载器时发生错误: {e}")
    else:
        print(f"  错误: 训练数据路径不存在: '{train_data_path}'")
        print("  请确认路径正确，并且数据集已按要求组织。")

    print("\n正在测试验证数据加载器 (val_loader)...")
    if os.path.exists(val_data_path):
        try:
            val_loader, val_dataset = get_flowers_dataloader(
                data_dir=val_data_path,
                batch_size=BATCH_SIZE_TEST,
                shuffle=False,
                is_train=False,
                num_workers=NUM_WORKERS_TEST
            )
            images, labels = next(iter(val_loader))
            print(f"  验证集批次图像形状: {images.shape}")
            print(f"  验证集批次标签形状: {labels.shape}")
            print(f"  验证集检测到的类别数量: {len(val_dataset.classes)}")
        except Exception as e:
            print(f"  测试验证数据加载器时发生错误: {e}")
    else:
        print(f"  错误: 验证数据路径不存在: '{val_data_path}'")

    # 你也可以类似地为 test_data_path 添加测试逻辑（如果 test 文件夹中有图像）
    # print("\n正在测试测试数据加载器 (test_loader)...")
    # if os.path.exists(test_data_path) and len(os.listdir(test_data_path)) > 0 : # 检查test文件夹是否为空
    #     try:
    #         # 假设测试集也使用与验证集相同的变换
    #         test_loader, test_dataset = get_flowers_dataloader(
    #             data_dir=test_data_path,
    #             batch_size=BATCH_SIZE_TEST,
    #             shuffle=False,
    #             is_train=False, # 通常测试集不进行数据增强
    #             num_workers=NUM_WORKERS_TEST
    #         )
    #         images, labels = next(iter(test_loader))
    #         print(f"  测试集批次图像形状: {images.shape}")
    #         print(f"  测试集批次标签形状: {labels.shape}")
    #         print(f"  测试集检测到的类别数量: {len(test_dataset.classes)}")
    #     except Exception as e:
    #         # 常见错误是 test 文件夹可能没有按类别分子文件夹，ImageFolder 会报错
    #         # 或者 test 文件夹本身就是空的，datasets.ImageFolder 在某些情况下会报错
    #         print(f"  测试测试数据加载器时发生错误: {e}")
    #         print(f"  请确保 '{test_data_path}' 目录非空且结构与 train/valid 类似 (每个类别一个子文件夹)，或者调整加载逻辑。")
    # else:
    #     print(f"  提示: 测试数据路径 '{test_data_path}' 不存在或为空，跳过测试加载。")