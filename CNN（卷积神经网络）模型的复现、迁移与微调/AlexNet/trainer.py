# alexnet_project/trainer/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json


class ModelTrainer:
    def __init__(self, model, dataloaders, dataset_sizes, config):
        self.model = model.to(config.DEVICE)
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.classifier[6].parameters(), lr=config.LEARNING_RATE)

        self.history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    def train(self):
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\n周期 {epoch}/{self.config.NUM_EPOCHS - 1}")
            print("-" * 10)

            # 训练和验证阶段
            for phase in ['train', 'valid']:
                epoch_loss, epoch_acc = self._run_phase(phase)

                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc)
                else:  # 验证阶段
                    self.history['valid_loss'].append(epoch_loss)
                    self.history['valid_acc'].append(epoch_acc)

                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                        torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                        print(f"新最佳模型已保存，验证准确率: {best_val_acc:.4f}")
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            self._save_history()
            epoch_time_elapsed = time.time() - epoch_start_time
            print(f"周期 {epoch} 在 {epoch_time_elapsed // 60:.0f}分 {epoch_time_elapsed % 60:.2f}秒 内完成")

            if epochs_no_improve >= self.config.PATIENCE:
                print(f"\n早停触发！连续 {self.config.PATIENCE} 个周期验证准确率未提升。")
                break

        print(f"\n训练完成。最佳验证准确率: {best_val_acc:.4f}")
        return self.history

    def _run_phase(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double().item() / self.dataset_sizes[phase]
        print(f"{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    def _save_history(self):
        with open(self.config.JSON_OUTPUT_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)