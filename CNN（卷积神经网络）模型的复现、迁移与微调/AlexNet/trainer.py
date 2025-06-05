"""
Trainer module: Manages the training and validation process for the flower classification model.
- Trains the model using a specified criterion, optimizer, and scheduler.
- Tracks training and validation metrics (loss and accuracy).
- Saves the best model weights based on validation accuracy.
"""
# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train(self):
        # 记录开始时间
        start_time = time.time()

        # 修正点 1：使用 .to() 将模型移动到指定设备
        self.model.to(self.device)
        print(f'Using device: {self.device}')

        # 深拷贝模型权重，用于保存最佳模型
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # 遍历所有训练周期
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            # 每个周期都包含训练和验证两个阶段
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                else:
                    self.model.eval()  # 设置模型为评估模式

                running_loss = 0.0
                running_corrects = 0

                # 遍历数据加载器中的数据
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 清零梯度
                    self.optimizer.zero_grad()

                    # 根据是否为训练阶段来决定是否计算梯度
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # 如果是训练阶段，则执行反向传播和优化
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # 累加损失和正确预测的数量
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # 在训练阶段结束后，更新学习率
                if phase == 'train':
                    self.scheduler.step()

                # 计算并保存当前周期的损失和准确率
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                if phase == 'train':
                    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())

                # 修正点 2：在验证阶段，如果发现更好的模型，则更新最佳准确率并保存其权重
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        # 修正点 3：训练结束后，添加总结信息和 return 语句

        # 计算总耗时
        time_elapsed = time.time() - start_time
        print(f'训练在 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒 内完成')
        print(f'最佳验证集准确率: {best_acc:4f}')

        # 加载在训练过程中找到的最佳模型权重
        self.model.load_state_dict(best_model_wts)

        # 返回训练好的模型和历史记录，以供后续使用
        return self.model, self.history