# src/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from model import FCNN # 假设你的模型文件是 model.py 并且类是 FCNN

class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,  # 验证加载器可选
                 criterion: nn.Module = None,
                 optimizer: optim.Optimizer = None,
                 device: str = 'cpu',
                 epochs: int = 10,
                 learning_rate: float = 0.001):
        """
        模型训练器。

        参数:
            model (nn.Module): 要训练的模型。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader, optional): 验证数据加载器。默认为 None。
            criterion (nn.Module, optional): 损失函数。如果为 None，则使用 CrossEntropyLoss。
            optimizer (optim.Optimizer, optional): 优化器。如果为 None，则使用 Adam。
            device (str): 训练设备 ('cuda' 或 'cpu')。
            epochs (int): 训练的总轮数。
            learning_rate (float): 优化器的学习率 (仅当 optimizer is None 时使用)。
        """
        self.model = model.to(device)  # 将模型移动到指定设备
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """计算准确率的辅助函数"""
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return 100 * correct / total

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # 乘以 batch size 以得到总损失

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:  # 每 100 个 batch 打印一次信息
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * correct_predictions / total_samples
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """在验证集上评估一个 epoch (如果提供了验证集)"""
        if self.val_loader is None:
            return None, None

        self.model.eval()  # 设置模型为评估模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # 在评估模式下不计算梯度
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * correct_predictions / total_samples
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc

    def train(self):
        """执行完整的训练过程"""
        print(f"开始在 {self.device} 上训练 {self.epochs} 个 epochs...")
        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            train_loss, train_acc = self.train_epoch()
            print(f"Epoch {epoch + 1} 完成: ")
            print(f"  训练损失 (Train Loss): {train_loss:.4f}, 训练准确率 (Train Acc): {train_acc:.2f}%")

            if self.val_loader:
                val_loss, val_acc = self.validate_epoch()
                if val_loss is not None and val_acc is not None:
                    print(f"  验证损失 (Val Loss): {val_loss:.4f}, 验证准确率 (Val Acc): {val_acc:.2f}%")

            # 你可以在这里添加保存模型的逻辑
            # 例如：if (epoch + 1) % 5 == 0: self.save_model(f"model_epoch_{epoch+1}.pth")

        print("\n训练完成!")
        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies
        }

    def save_model(self, path: str):
        """保存模型状态字典"""
        print(f"保存模型到 {path}...")
        torch.save(self.model.state_dict(), path)

# 注意：下面的 __main__ 块通常不会放在 trainer.py 中。
# 训练的启动通常在 main.py 中完成，它会导入 DataLoader, Model, 和 Trainer。
# 这里只是为了演示如何实例化和使用 Trainer。
#
# if __name__ == '__main__':
#     # 这是一个非常简化的演示，实际使用时应在 main.py 中组织
#     from data_loader import MNISTDataLoaders # 假设可以正确导入
#     from model import FCNN                   # 假设可以正确导入
#
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     EPOCHS = 5 # 仅用于测试
#     BATCH_SIZE = 128
#     LEARNING_RATE = 0.001
#
#     # 1. 加载数据
#     # 注意路径，因为此文件在 src 下，所以 ../data
#     data_loaders = MNISTDataLoaders(data_dir='../data', batch_size=BATCH_SIZE)
#     train_loader = data_loaders.get_train_loader()
#     test_loader = data_loaders.get_test_loader() # 我们将测试集用作验证集进行演示
#
#     # 2. 初始化模型
#     # FCNN 默认 input_size=784 (因为它内部有 Flatten), num_classes=10
#     mnist_model = FCNN().to(DEVICE)
#
#     # 3. 初始化训练器
#     trainer = ModelTrainer(
#         model=mnist_model,
#         train_loader=train_loader,
#         val_loader=test_loader, # 用测试集作为验证集
#         device=DEVICE,
#         epochs=EPOCHS,
#         learning_rate=LEARNING_RATE
#     )
#
#     # 4. 开始训练
#     training_history = trainer.train()
#
#     # 5. (可选) 保存最终模型
#     # trainer.save_model("../saved_models/mnist_fcnn_final.pth") # 注意路径
#
#     print("\n训练历史:")
#     print(training_history)