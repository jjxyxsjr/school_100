# predict_custom_unet.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 从我们自己新建的、带有CBAM的U-Net文件导入模型
from unet_with_cbam import UNet
# 数据加载部分保持不变
from CNN的应用_目标检测_语义分割.语义分割.pets_dataset import PetsDataset, pet_transform

# --- Matplotlib 中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# -----------------------------

# --- 1. 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/pets"
# 确保这里的模型路径与您训练时保存的路径一致
MODEL_PATH = "original_unet_with_cbam_256px.pth"
BATCH_SIZE_PREDICT = 4  # 可视化时不需要太大的batch size
IMAGE_SIZE_PREDICT = (256, 256)  # 确保与训练时一致

# --- 2. 定义模型 (必须与训练时完全一致) ---
print("初始化模型结构...")
# use_cbam_everywhere 参数需要与训练时一致
model = UNet(n_channels=3, n_classes=1, bilinear=True, use_cbam_everywhere=True).to(DEVICE)

# --- 3. 加载已训练好的模型权重 ---
if os.path.exists(MODEL_PATH):
    print(f"加载模型权重: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
else:
    print(f"错误: 模型文件 {MODEL_PATH} 未找到。请先运行训练脚本。")
    exit()

model.eval()  # 设置为评估模式

# --- 4. 加载一些数据用于可视化 ---
print("加载数据...")
# 确保 pet_transform 使用了与 IMAGE_SIZE_PREDICT 一致的 target_size
# (在 pets_dataset.py 中修改 pet_transform 的 target_size)
# 为了演示，我们只取数据集中的前几个样本
# 您也可以像训练脚本一样划分验证集
try:
    # 尝试加载整个数据集，然后取一个子集
    full_dataset_predict = PetsDataset(
        image_dir=os.path.join(DATA_PATH, "images"),
        mask_dir=os.path.join(DATA_PATH, "annotations"),
        transform=pet_transform  # pet_transform 内部会使用 IMAGE_SIZE_PREDICT
    )
    # 如果数据集很大，可以只取一个小的子集进行预测
    # from torch.utils.data import Subset
    # indices = torch.randperm(len(full_dataset_predict)).tolist()[:BATCH_SIZE_PREDICT*2] # 取一些随机样本
    # predict_dataset = Subset(full_dataset_predict, indices)
    # loader_predict = DataLoader(predict_dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False)

    # 或者直接用整个验证集（如果之前划分了的话）
    # _, val_dataset_predict = random_split(full_dataset_predict, [train_size, val_size]) # 需要train_size, val_size
    # loader_predict = DataLoader(val_dataset_predict, batch_size=BATCH_SIZE_PREDICT, shuffle=False)

    # 这里为了简单，我们直接用整个数据集然后取一个批次
    loader_predict = DataLoader(full_dataset_predict, batch_size=BATCH_SIZE_PREDICT, shuffle=True)

except FileNotFoundError:
    print(f"错误: 数据集路径 {DATA_PATH} 未找到或不完整。")
    exit()

# --- 5. 进行预测和可视化 ---
print("进行预测并显示结果...")
try:
    images, masks = next(iter(loader_predict))
except StopIteration:
    print("错误：数据加载器为空，无法获取数据进行预测。请检查数据集和批次大小。")
    exit()

images = images.to(DEVICE)

with torch.no_grad():
    preds_logits = model(images)
    preds_prob = torch.sigmoid(preds_logits)
    preds_binary = (preds_prob > 0.5).float()  # 二值化预测结果

preds_binary_cpu = preds_binary.cpu()
images_cpu = images.cpu()
masks_cpu = masks.cpu()  # 真实掩膜也移到CPU

# 显示结果
num_images_to_show = min(BATCH_SIZE_PREDICT, images_cpu.size(0))  # 确保不超过实际加载的图片数量
fig, axes = plt.subplots(num_images_to_show, 3, figsize=(12, num_images_to_show * 4))
if num_images_to_show == 1:  # 如果只显示一张图，axes不是数组
    axes = axes.reshape(1, -1)
fig.suptitle("模型预测结果 (U-Net原型+CBAM)", fontsize=16)

for i in range(num_images_to_show):
    # 原始图像
    ax = axes[i, 0]
    # permute(1, 2, 0) 将 CHW 转换为 HWC 以便matplotlib显示
    ax.imshow(images_cpu[i].permute(1, 2, 0))
    ax.set_title(f"原始图像 {i + 1}")
    ax.axis('off')

    # 真实掩膜
    ax = axes[i, 1]
    ax.imshow(masks_cpu[i].squeeze(), cmap='gray')  # squeeze() 移除单通道维度
    ax.set_title(f"真实掩膜 {i + 1}")
    ax.axis('off')

    # 预测掩膜
    ax = axes[i, 2]
    ax.imshow(preds_binary_cpu[i].squeeze(), cmap='gray')
    ax.set_title(f"预测掩膜 {i + 1}")
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局防止标题重叠
plt.show()

print("\n预测和可视化完成。")