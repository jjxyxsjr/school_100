# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from 任务二.GooLeNet import config


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # 重新加入 history 用于绘图
        self.history = {
            's1_train_loss': [], 's1_val_loss': [], 's1_train_acc': [], 's1_val_acc': [],
            's2_train_loss': [], 's2_val_loss': [], 's2_train_acc': [], 's2_val_acc': [],
            'early_stop_epoch': None
        }

    def _run_epoch(self, dataloader, is_training):
        self.model.train(is_training)
        pbar_desc = "训练中" if is_training else "评估中"

        total_loss, correct_predictions, total_samples = 0.0, 0, 0
        pbar = tqdm(dataloader, desc=pbar_desc, leave=False)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            pbar.set_postfix(loss=total_loss / total_samples, acc=correct_predictions.double().item() / total_samples)

        return total_loss / total_samples, (correct_predictions.double() / total_samples).item()

    def train_stage1(self, train_loader, val_loader):
        print("\n--- 开始第一阶段训练: 仅训练全连接层 ---")
        self.optimizer = optim.Adam(
            self.model.classifier[6].parameters(),
            lr=config.STAGE1_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        early_stopper = EarlyStopper(patience=config.PATIENCE)
        best_val_loss = float('inf')

        for epoch in range(1, config.STAGE1_EPOCHS + 1):
            print(f"\nStage 1 - Epoch {epoch}/{config.STAGE1_EPOCHS}:")
            train_loss, train_acc = self._run_epoch(train_loader, is_training=True)
            val_loss, val_acc = self._run_epoch(val_loader, is_training=False)

            # 记录历史数据
            self.history['s1_train_loss'].append(train_loss)
            self.history['s1_train_acc'].append(train_acc)
            self.history['s1_val_loss'].append(val_loss)
            self.history['s1_val_acc'].append(val_acc)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), config.STAGE1_MODEL_PATH)
                self.history['early_stop_epoch'] = epoch  # 记录最佳 epoch
                print(f"  -> 验证Loss改善，模型已保存至 {config.STAGE1_MODEL_PATH}")

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"  Early stopping 触发于 epoch {epoch}.")
                break

        print(f"\n加载第一阶段最佳模型 (来自 Epoch {self.history['early_stop_epoch']})...")
        self.model.load_state_dict(torch.load(config.STAGE1_MODEL_PATH))

    def train_stage2(self, train_loader, val_loader):
        print("\n--- 开始第二阶段训练: 微调部分解冻层 ---")
        print("解冻模型分类器层 (model.classifier)...")
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=config.STAGE2_LR, weight_decay=config.WEIGHT_DECAY)
        best_val_loss_s2 = float('inf')

        for epoch in range(1, config.STAGE2_EPOCHS + 1):
            print(f"\nStage 2 - Epoch {epoch}/{config.STAGE2_EPOCHS}:")
            train_loss, train_acc = self._run_epoch(train_loader, is_training=True)
            val_loss, val_acc = self._run_epoch(val_loader, is_training=False)

            # 记录历史数据
            self.history['s2_train_loss'].append(train_loss)
            self.history['s2_train_acc'].append(train_acc)
            self.history['s2_val_loss'].append(val_loss)
            self.history['s2_val_acc'].append(val_acc)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            if val_loss < best_val_loss_s2:
                best_val_loss_s2 = val_loss
                torch.save(self.model.state_dict(), config.STAGE2_MODEL_PATH)
                print(f"  -> S2 验证Loss改善，模型已保存至 {config.STAGE2_MODEL_PATH}")

        print(f"\n加载第二阶段最佳模型...")
        self.model.load_state_dict(torch.load(config.STAGE2_MODEL_PATH))

    def test(self, test_loader, stage_name):
        print(f"\n--- 在测试集上评估 {stage_name} 模型 ---")
        test_loss, test_acc = self._run_epoch(test_loader, is_training=False)
        print(f"测试结果 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        return test_acc