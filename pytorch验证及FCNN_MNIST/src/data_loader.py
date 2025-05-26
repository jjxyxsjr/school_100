# src/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTDataLoaders:
    def __init__(self, data_dir: str = './data', batch_size: int = 64, train_shuffle: bool = True,
                 test_shuffle: bool = False):
        """
        初始化 MNIST 数据加载器。

        参数:
            data_dir (str): MNIST 数据集存储的目录路径。
            batch_size (int): 每个批次的样本数量。
            train_shuffle (bool): 是否在每个 epoch 开始时打乱训练数据。
            test_shuffle (bool): 是否在每个 epoch 开始时打乱测试数据。
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle

        # 定义数据转换
        # MNIST 数据集的均值和标准差 (单通道)
        # 这些值是根据 MNIST 训练集计算得出的常用值
        mean = 0.1307
        std = 0.3081

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将 PIL 图像或 NumPy ndarray 转换为 FloatTensor，并将像素值缩放到 [0.0, 1.0]
            transforms.Normalize((mean,), (std,)),  # 使用均值和标准差进行归一化
            # 注意：FCNN 通常需要展平的输入。
            # 你可以在这里添加 transforms.Flatten()，或者在模型的第一层进行展平。
            # 如果在这里添加：transforms.Flatten()
        ])

        # 下载/加载训练数据集
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,  # 如果数据不存在，则下载
            transform=self.transform
        )

        # 下载/加载测试数据集
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_train_loader(self) -> DataLoader:
        """
        获取训练数据加载器。
        """
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=0  # Windows 下建议为0，Linux/macOS可以适当调大
        )
        return train_loader

    def get_test_loader(self) -> DataLoader:
        """
        获取测试数据加载器。
        """
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,  # 测试时通常也可以用较大的batch_size
            shuffle=self.test_shuffle,
            num_workers=0
        )
        return test_loader


if __name__ == '__main__':
    # 这个 __main__ 块可以用来测试你的数据加载器是否工作正常

    # 实例化数据加载器
    mnist_loaders = MNISTDataLoaders(data_dir='../data', batch_size=128)  # 注意路径，因为此文件在 src 下

    # 获取训练和测试加载器
    train_loader = mnist_loaders.get_train_loader()
    test_loader = mnist_loaders.get_test_loader()

    print(f"训练集样本数量: {len(train_loader.dataset)}")
    print(f"测试集样本数量: {len(test_loader.dataset)}")

    # 检查一个批次的数据
    try:
        train_features, train_labels = next(iter(train_loader))
        print(f"\n一个训练批次的数据形状:")
        print(f"特征 (Features) batch shape: {train_features.size()}")  # 应该是 [batch_size, 1, 28, 28]
        print(f"标签 (Labels) batch shape: {train_labels.size()}")  # 应该是 [batch_size]
        print(f"第一个样本的标签: {train_labels[0]}")

        # 如果你没有在 transform 中使用 Flatten，可以在这里手动检查
        # flattened_features = train_features.view(train_features.size(0), -1)
        # print(f"展平后的特征 batch shape: {flattened_features.size()}") # 应该是 [batch_size, 784]

    except Exception as e:
        print(f"在获取数据批次时发生错误: {e}")
        print("请确保 MNIST 数据已下载到指定的 'data_dir' 目录中，并且目录结构正确。")
        print("通常的结构是: ./data/MNIST/raw/train-images-idx3-ubyte 等文件。")

    try:
        test_features, test_labels = next(iter(test_loader))
        print(f"\n一个测试批次的数据形状:")
        print(f"特征 (Features) batch shape: {test_features.size()}")
        print(f"标签 (Labels) batch shape: {test_labels.size()}")
        print(f"第一个样本的标签: {test_labels[0]}")
    except Exception as e:
        print(f"在获取测试数据批次时发生错误: {e}")