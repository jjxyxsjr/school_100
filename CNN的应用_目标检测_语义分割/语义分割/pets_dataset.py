# pets_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np  # 需要导入 numpy


class PetsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 确保只加载.jpg文件，避免其他文件干扰
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Oxford-IIIT Pet 数据集的掩膜在 'annotations/trimaps' 子目录下
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, 'trimaps', mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 以灰度模式加载掩膜

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def pet_transform(image, mask):
    # 1. 调整图像和掩膜的大小
    # 对于U-Net原型，我们可以使用稍小一些的分辨率，例如128x128或256x256
    # 如果您的数据集比较简单，128x128可能就足够了
    target_size = (128, 128)
    resize_transform = transforms.Resize(target_size)
    image = resize_transform(image)
    mask = resize_transform(mask)

    # 2. 将PIL图像转换为Tensor (只对 image 操作)
    image = transforms.functional.to_tensor(image)

    # 3. 处理掩膜 (在Tensor转换前处理)
    # 将掩膜转换为NumPy数组，方便用整数值进行操作
    mask = np.array(mask)
    # 原始掩膜: 1=前景(宠物), 2=背景, 3=轮廓(也视为前景)

    # 创建一个新的掩膜数组，初始值为0 (背景)
    processed_mask = np.zeros(mask.shape, dtype=np.uint8)
    # 将前景(1)和轮廓(3)的区域设置为1
    processed_mask[mask == 1] = 1
    processed_mask[mask == 3] = 1
    # 背景(2)的区域保持为0 (因为默认就是0)

    # 4. 将处理好的 NumPy 掩膜数组转换为 PyTorch 张量
    # 注意这里要用 from_numpy，并且要增加一个通道维度，并转换为浮点型
    mask = torch.from_numpy(processed_mask).unsqueeze(0).float()

    return image, mask