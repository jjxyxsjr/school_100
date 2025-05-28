from torchvision import transforms
from trainer import CatDogTrainer

# 图像预处理流程
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224，VGG网络输入尺寸要求
    transforms.ToTensor(),           # 转为Tensor
])

# 训练和测试数据路径，确保文件夹存在（即使为空也不会报错）
train_dir = "./data/train"
test_dir = "./data/test"

if __name__ == "__main__":
    skip_train = True  # True：检测到模型权重文件时跳过训练，直接加载测试；False：正常训练
    trainer = CatDogTrainer(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        skip_train=skip_train
    )
    trainer.run()
# Training completed in 827.1 seconds

# ✅ 已加载模型权重：./checkpoints/catdog_vgg16.pth
# 测试集准确率: 96.44% | 平均损失: 0.0944



# D:\Anaconda\envs\DL\python.exe D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\VGG\catdog_classifier\main.py
# 数据加载完毕：训练集图像 8005 张，测试集图像 2023 张。
# ℹ️ 未找到已保存模型，将重新训练。
# 警告：设置了skip_train=True，但无法加载模型或模型不存在。将开始新的训练。
# 开始在 cuda 上进行训练，共 10 个 epochs...
#
# --- Epoch 10/10 总结 ---
# 平均训练损失: 0.0735
# 平均测试损失: 0.1138 | 测试准确率: 96.34%
# ------------------------------
# 训练完成，总耗时: 1028.9 秒 (17.15 分钟)
# ✅ 模型已保存至 ./checkpoints/catdog_vgg16.pth
#
# --- 训练完成后的最终评估 ---
# 最终训练集评估集准确率: 98.25% | 平均损失: 0.0421
# 最终测试集评估集准确率: 96.34% | 平均损失: 0.1138
# 📈 损失曲线图已保存至 epoch_loss_curves.png
#
# 进程已结束，退出代码为 0
