import json
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open(r"D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\training_history_feature_extract.json", "r", encoding="utf-8") as f:
    history = json.load(f)


# 提取数据
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_acc = history["train_acc"]
val_acc = history["val_acc"]
epochs = range(1, len(train_loss) + 1)

# 设置画布
plt.figure(figsize=(12, 5))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

