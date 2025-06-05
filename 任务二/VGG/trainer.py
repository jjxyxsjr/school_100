# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from collections import defaultdict

import config
from utils import plot_training_results


class Trainer:
    def __init__(self, model, dataloaders, dataset_sizes, device):
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def _train_epoch(self, optimizer):
        """一个 epoch 的训练逻辑"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.dataloaders['train']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.dataset_sizes['train']
        epoch_acc = running_corrects.double() / self.dataset_sizes['train']
        return epoch_loss, epoch_acc.item()

    def _validate_epoch(self):
        """一个 epoch 的验证逻辑"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.dataloaders['validation']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.dataset_sizes['validation']
        epoch_acc = running_corrects.double() / self.dataset_sizes['validation']
        return epoch_loss, epoch_acc.item()

    def train(self, stage_num, epochs, lr, weight_decay, model_save_path, visualization_path, load_path=None):
        """
        通用的训练循环，包含 Early Stopping。
        """
        print(f"\n--- 开始训练阶段 {stage_num} ---")
        print(f"Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")

        if load_path:
            self.model.load_state_dict(torch.load(load_path))
            print(f"已从 {load_path} 加载模型权重。")

        # 根据阶段配置优化器
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                               weight_decay=weight_decay)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_val_loss = float('inf')
        epochs_no_improve = 0
        history = defaultdict(list)

        since = time.time()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            train_loss, train_acc = self._train_epoch(optimizer)
            val_loss, val_acc = self._validate_epoch()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'训练 Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'验证 Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            # Early Stopping 和保存最佳模型逻辑
            # 我们选择在验证集上损失最低的模型为最佳模型
            if val_loss < best_val_loss:
                print(f"验证集损失从 {best_val_loss:.4f} 降低到 {val_loss:.4f}。保存模型...")
                best_val_loss = val_loss
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), model_save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"验证集损失没有改善，计数: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")

            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print("触发 Early Stopping!")
                break

        time_elapsed = time.time() - since
        print(f'训练阶段 {stage_num} 完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
        print(f'最佳验证集 Acc: {best_acc:4f}')

        # 加载最佳模型权重以供后续使用
        self.model.load_state_dict(best_model_wts)

        # 可视化
        plot_training_results(history, stage_num, visualization_path)

        return self.model

    def test(self):
        """在测试集上评估模型性能"""
        print("\n--- 在测试集上评估最终模型 ---")
        self.model.eval()
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / self.dataset_sizes['test']
        print(f'测试集上的准确率: {test_acc:.4f}')
        return test_acc.item()