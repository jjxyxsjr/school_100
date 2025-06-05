# train_custom_unet.py (已修正警告)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import os

# 从我们自己新建的、带有CBAM的U-Net文件导入模型
from unet_with_cbam import UNet
# 数据加载部分保持不变
from CNN的应用_目标检测_语义分割.语义分割.pets_dataset import PetsDataset, pet_transform


# 损失函数类保持不变
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs_prob = torch.sigmoid(inputs)
        inputs_prob_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_prob_flat * targets_flat).sum()
        dice_denominator = inputs_prob_flat.sum() + targets_flat.sum()
        dice_loss = 1 - (2. * intersection + smooth) / (dice_denominator + smooth)
        combined_loss = bce_loss + dice_loss
        return combined_loss


if __name__ == '__main__':
    # --- 1. 超参数和配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    DATA_PATH = "data/pets"
    MODEL_SAVE_PATH = "original_unet_with_cbam_128px.pth"
    IMAGE_SIZE = (128, 128)

    # --- 2. 加载数据 ---
    print("加载数据中...")
    full_dataset = PetsDataset(
        image_dir=os.path.join(DATA_PATH, "images"),
        mask_dir=os.path.join(DATA_PATH, "annotations"),
        transform=pet_transform
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"数据加载完成。训练集: {train_size}, 验证集: {val_size}, 图像尺寸: {IMAGE_SIZE}")

    # --- 3. 初始化模型、损失函数和优化器 ---
    print("初始化U-Net原型+CBAM模型...")
    model = UNet(n_channels=3, n_classes=1, bilinear=True, use_cbam_everywhere=True).to(DEVICE)

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # #################### 警告修正之处 (第1步) ####################
    # 删除了 verbose=True 参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    # ##########################################################

    # --- 4. 训练循环 ---
    print(f"开始在 {DEVICE} 上训练模型: {MODEL_SAVE_PATH}...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * images.size(0)

        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item() * images.size(0)

        avg_epoch_val_loss = epoch_val_loss / len(val_loader.dataset)

        # #################### 警告修正之处 (第2步) ####################
        # 手动获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 在打印信息中加入当前学习率
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
              f"训练损失: {avg_epoch_train_loss:.4f}, "
              f"验证损失: {avg_epoch_val_loss:.4f}, "
              f"当前学习率: {current_lr:.6f}")
        # ##########################################################

        scheduler.step(avg_epoch_val_loss)

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  验证损失降低，模型已保存至 {MODEL_SAVE_PATH}")

    print("\n训练完成。")