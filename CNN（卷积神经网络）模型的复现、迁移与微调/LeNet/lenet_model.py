import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernLeNetFCNTanh(nn.Module):  # 更改类名以反映 Tanh 激活
    """
    一个现代化、全卷积的 LeNet 风格架构，适用于 CIFAR-10，
    现在使用 Tanh 激活函数。
    - 使用 3 个输入通道（用于 RGB 图像）。
    - 采用 Tanh 激活函数。
    - 使用 MaxPool2d 进行下采样。
    - F5 层是一个卷积层。
    - 最后一层输出 10 个类别。
    """

    def __init__(self):
        super(ModernLeNetFCNTanh, self).__init__()

        # 第一个卷积层 (C1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

        # 第一个池化层 (S2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层 (C3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 第二个池化层 (S4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # F5 作为卷积层
        # S4 层的输出特征图尺寸是 16x5x5。
        # kernel_size=5 确保这个卷积的输出是 1x1。
        self.conv_f5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # 全连接层 (F6)
        # conv_f5 的输出是 120x1x1。连接到全连接层前，需要将其展平为 120。
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        # 输出层
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 应用 conv1，然后 Tanh 激活，然后 pool1
        x = self.pool1(F.tanh(self.conv1(x)))  # 从 F.relu 改为 F.tanh

        # 应用 conv2，然后 Tanh 激活，然后 pool2
        x = self.pool2(F.tanh(self.conv2(x)))  # 从 F.relu 改为 F.tanh

        # 应用 conv_f5 (F5 作为卷积层)，Tanh 激活
        x = F.tanh(self.conv_f5(x))  # 从 F.relu 改为 F.tanh

        # 将特征图展平以连接到全连接层
        x = x.view(-1, 120)

        # 应用 fc2，然后 Tanh 激活
        x = F.tanh(self.fc2(x))  # 从 F.relu 改为 F.tanh

        # 应用 fc3 (输出层)，此处不加激活函数
        x = self.fc3(x)
        return x