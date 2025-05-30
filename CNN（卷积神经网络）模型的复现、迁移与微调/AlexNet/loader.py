# alexnet_project/data_loader/loader.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_data_loaders(dataset_path, batch_size):
    """准备并返回训练和验证的DataLoader"""

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }

    try:
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x])
            for x in ['train', 'valid']
        }
    except FileNotFoundError:
        print(f"错误：在 {dataset_path} 找不到数据集")
        return None, None

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'valid']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"数据加载完成。训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['valid']}")

    return dataloaders, dataset_sizes