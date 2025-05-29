import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
# 导入新的 ModernLeNetFCNTanh 类
from lenet_model import ModernLeNetFCNTanh


class LeNetTrainer:
    def __init__(self, data_root='./data', batch_size=64, test_batch_size=1000, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr

        self._load_data(data_root)
        self._initialize_model()

    def _load_data(self, data_root):
        """
        加载 CIFAR-10 数据集 (RGB)。
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])

        self.train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        self.test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        print("数据集加载成功 (CIFAR-10 RGB)。")

    def _initialize_model(self):
        # 实例化新的 ModernLeNetFCNTanh 模型
        self.model = ModernLeNetFCNTanh().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print(f"模型初始化并运行在设备: {self.device}")
        # print("模型架构:")
        # print(self.model) # 取消注释以查看详细的模型结构

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        end_time = time.time()
        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch {epoch} - 训练损失: {avg_loss:.4f}, 时间: {end_time - start_time:.2f} 秒")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        end_time = time.time()
        accuracy = 100. * correct / total
        print(f"测试准确率: {accuracy:.2f}%, 时间: {end_time - start_time:.2f} 秒")
        return accuracy

    def save_model(self, path='lenet_model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")

    def load_model(self, path='lenet_model.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 {path} 加载。")

    def run(self, num_epochs=3, output_dir='results', output_filename='lenet_output_modern_fcn_tanh.txt'):
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

        original_stdout = None
        try:
            with open(output_filepath, 'w') as f:
                original_stdout = os.dup(1)
                os.dup2(f.fileno(), 1)

                print(f"--- 现代化全卷积 LeNet (Tanh 激活) 模型训练开始 ---")
                print(f"参数: 批次大小={self.batch_size}, 测试批次大小={self.test_batch_size}, 学习率={self.lr}")
                print(f"数据集: CIFAR-10 (RGB)")

                for epoch in range(1, num_epochs + 1):
                    self.train_epoch(epoch)
                    self.evaluate()
                print(f"--- 现代化全卷积 LeNet (Tanh 激活) 模型训练结束 ---")

        finally:
            if original_stdout is not None:
                os.dup2(original_stdout, 1)
                os.close(original_stdout)

        print(f"\n训练和测试结果已保存到 '{output_filepath}'")


if __name__ == "__main__":
    trainer = LeNetTrainer()
    trainer.run(num_epochs=50)

    model_save_path = os.path.join('results', 'trained_modern_lenet_fcn_tanh_cifar10.pth')
    trainer.save_model(model_save_path)