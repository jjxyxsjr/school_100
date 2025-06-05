# unet_with_cbam.py

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# #################### CBAM 模块的实现 ####################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保卷积核大小是奇数
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_cat)
        return self.sigmoid(x_att)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_ca = self.ca(x) * x  # 通道注意力
        x_out = self.sa(x_ca) * x_ca  # 空间注意力
        return x_out


# #######################################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 => CBAM (可选)"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_cbam=True):  # 添加 use_cbam 开关
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        x_conv = self.double_conv(x)
        if self.use_cbam:
            x_att = self.cbam(x_conv)
            return x_att
        else:
            return x_conv


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_cbam_in_doubleconv=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_cbam=use_cbam_in_doubleconv)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_cbam_in_doubleconv=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 注意：这里的 in_channels 是拼接后的通道数，所以 DoubleConv 的输入是 in_channels
            # 而不是像原始U-Net那样是 in_channels // 2 (如果用转置卷积的话)
            # 但因为我们先上采样再拼接，所以 DoubleConv 的输入应该是 x2 的通道数 + x1 上采样后的通道数
            # x1 上采样后通道数是 in_channels // 2 (因为 in_channels 是 x1 和 x2 拼接前的总通道数)
            # x2 的通道数也是 in_channels // 2
            # 所以拼接后是 in_channels，DoubleConv 的输入是 in_channels
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_cbam=use_cbam_in_doubleconv)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_cbam=use_cbam_in_doubleconv)

        self.bilinear = bilinear

    def forward(self, x1, x2):  # x1 是上采样路径的输入, x2 是跳跃连接的输入
        if self.bilinear:
            x1 = self.up(x1)
        else:  # 如果不是双线性插值，则 x1 已经是通过转置卷积上采样过的
            # 此时 x1 的通道数是 in_channels // 2
            pass  # x1 已经是上采样后的了

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_cbam_everywhere=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(n_channels, 64, use_cbam=use_cbam_everywhere)
        self.down1 = Down(64, 128, use_cbam_in_doubleconv=use_cbam_everywhere)
        self.down2 = Down(128, 256, use_cbam_in_doubleconv=use_cbam_everywhere)
        self.down3 = Down(256, 512, use_cbam_in_doubleconv=use_cbam_everywhere)

        # 根据是否使用双线性插值，确定中间层的通道数
        # 如果是转置卷积，通道数会减半；如果是双线性插值，通道数不变，由后续卷积调整
        factor = 2 if bilinear else 1
        # 实际上，如果用转置卷积，down4的输入是512，输出是512 (in_channels // 2)
        # 如果用双线性插值，down4的输入是512，输出是1024
        # 为了简化，我们让 down4 的输出通道数固定，然后在 Up 模块中处理

        self.down4 = Down(512, 1024 // factor, use_cbam_in_doubleconv=use_cbam_everywhere)

        # 解码器
        # Up 模块的 in_channels 是指跳跃连接的特征图通道数 + 上一层上采样后的特征图通道数
        # 例如 up1: x5 (1024/factor) + x4 (512)
        # 如果 bilinear=True, factor=2, x5通道是512, x4通道是512, 拼接后是1024
        # 如果 bilinear=False, factor=1, x5通道是1024, 转置卷积后是512, x4通道是512, 拼接后是1024
        self.up1 = Up(1024, 512 // factor, bilinear, use_cbam_in_doubleconv=use_cbam_everywhere)
        self.up2 = Up(512, 256 // factor, bilinear, use_cbam_in_doubleconv=use_cbam_everywhere)
        self.up3 = Up(256, 128 // factor, bilinear, use_cbam_in_doubleconv=use_cbam_everywhere)
        self.up4 = Up(128, 64, bilinear, use_cbam_in_doubleconv=use_cbam_everywhere)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # 输出 64通道
        x2 = self.down1(x1)  # 输出 128通道
        x3 = self.down2(x2)  # 输出 256通道
        x4 = self.down3(x3)  # 输出 512通道
        x5 = self.down4(x4)  # 输出 1024/factor 通道 (bilinear时512, 转置卷积时1024)

        # 解码器路径
        # up1: 输入 x5 和 x4
        # 如果 bilinear=True, x5(512), x4(512) -> cat(1024) -> conv -> 512/factor (256)
        # 如果 bilinear=False, x5(1024) -> transpose_conv(512), x4(512) -> cat(1024) -> conv -> 512/factor (512)
        # 为了统一，我们应该让 Up 模块的第一个参数是 x5 和 x4 拼接后的通道数
        # 而不是像原始U-Net那样只给 x5 的通道数。
        # 或者，更简单的方式是，Up模块的第一个参数是上一级解码器输出的通道数，
        # 然后在Up模块内部处理与编码器跳跃连接的拼接。
        # 这里的 Up 模块设计是：第一个参数是上采样路径的输入通道数，第二个是跳跃连接的通道数
        # 但实际上，smp库的Unet实现中，Up的第一个参数是上一层输出的通道数，第二个参数是这一层要输出的通道数
        # 我们这里保持我们之前的实现逻辑：

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits