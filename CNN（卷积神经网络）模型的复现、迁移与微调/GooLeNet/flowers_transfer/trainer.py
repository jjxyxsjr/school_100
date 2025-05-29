# flowers_transfer/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import os
import time
import copy  # 用于深拷贝模型权重
from tqdm import tqdm  # 用于显示进度条

# 导入配置和模型获取函数
import config


# (我们稍后会在 run/*.py 脚本中创建模型和优化器实例，然后传递给 Trainer)
# from model import get_googlenet_model (不需要在这里导入，模型会作为参数传入)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                 epochs, checkpoint_dir, model_name="best_model.pth"):
        """
        Trainer类的构造函数。

        参数:
            model (torch.nn.Module): 要训练的模型。
            train_loader (torch.utils.data.DataLoader): 训练数据加载器。
            val_loader (torch.utils.data.DataLoader): 验证数据加载器。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。
            device (torch.device): 训练设备 (e.g., 'cuda' or 'cpu')。
            epochs (int): 训练的总周期数。
            checkpoint_dir (str): 保存模型权重的目录。
            model_name (str): 保存最佳模型的文件名。
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.best_model_path = os.path.join(checkpoint_dir, model_name)

        # 确保模型在正确的设备上
        self.model.to(self.device)

        # 用于记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0.0  # 用于跟踪最佳验证准确率

    def _train_one_epoch(self):
        """执行一个训练周期。"""
        self.model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 使用 tqdm 创建进度条
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]",
                            leave=False)

        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            # GoogLeNet 在训练模式且启用 aux_logits 时，会返回一个包含主输出和辅助输出的元组
            # (GoogLeNetOutputs(logits, aux_logits1, aux_logits2))
            if self.model.training and hasattr(self.model, 'aux_logits') and self.model.aux_logits:
                outputs, aux1_outputs, aux2_outputs = self.model(inputs)
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux1_outputs, labels)
                loss3 = self.criterion(aux2_outputs, labels)
                loss = loss1 + 0.3 * loss2 + 0.3 * loss3  # GoogLeNet 的标准损失计算
            else:  # eval 模式或 aux_logits 关闭
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)

            # 更新进度条的后缀信息
            progress_bar.set_postfix(loss=loss.item(),
                                     acc=correct_predictions / total_samples if total_samples > 0 else 0)

        progress_bar.close()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def _validate_one_epoch(self):
        """执行一个验证周期。"""
        self.model.eval()  # 设置模型为评估模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Valid]", leave=False)

        with torch.no_grad():  # 在验证阶段不计算梯度
            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 前向传播 (在 eval 模式下，GoogLeNet 通常只返回主输出)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data).item()
                total_samples += labels.size(0)

                progress_bar.set_postfix(loss=loss.item(),
                                         acc=correct_predictions / total_samples if total_samples > 0 else 0)

        progress_bar.close()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def train(self):
        """执行完整的训练过程。"""
        print(f"开始训练，将在 {self.device} 上运行 {self.epochs} 个周期。")
        print(f"模型将保存在: {self.best_model_path}")

        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self._train_one_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # 验证
            val_loss, val_acc = self._validate_one_epoch()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # ReduceLROnPlateau 需要监控指标
                else:
                    self.scheduler.step()  # 其他调度器如 StepLR, CosineAnnealingLR

            epoch_duration = time.time() - epoch_start_time

            print(f"周期 {epoch + 1}/{self.epochs} | "
                  f"耗时: {epoch_duration:.2f}s | "
                  f"学习率: {current_lr:.1e} | "
                  f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
                  f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(self.best_model_path, epoch, val_acc, is_best=True)
                print(f"  验证准确率提升 ({self.best_val_acc:.4f})。保存模型到 {self.best_model_path}")

            # (可选) 定期保存检查点
            # if (epoch + 1) % 5 == 0: # 每5个周期保存一次
            #     periodic_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            #     self.save_checkpoint(periodic_checkpoint_path, epoch, val_acc)
            #     print(f"  已保存周期性检查点到 {periodic_checkpoint_path}")

        total_training_time = time.time() - start_time
        print(f"\n训练完成。总耗时: {total_training_time // 60:.0f}分 {total_training_time % 60:.0f}秒")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")

        # (可选) 加载最佳模型权重以供后续使用或评估
        # self.load_checkpoint(self.best_model_path)
        # print(f"已加载最佳模型权重从 {self.best_model_path}")

        return self.history

    def save_checkpoint(self, filepath, epoch, val_accuracy, is_best=False):
        """保存模型检查点。"""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,  # 或者 val_accuracy 如果你想保存当前检查点的准确率
            'history': self.history  # 保存到目前为止的训练历史
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # 如果是最佳模型，通常只保存模型权重以减小文件大小
        # 但为了演示，我们保存完整状态
        if is_best:
            torch.save(state, filepath)  # 直接覆盖最佳模型
        else:  # 对于周期性检查点，可以使用不同的文件名
            torch.save(state, filepath)

    def load_checkpoint(self, filepath):
        """加载模型检查点。"""
        if not os.path.exists(filepath):
            print(f"检查点文件未找到: {filepath}")
            return None

        checkpoint = torch.load(filepath, map_location=self.device)  # 确保加载到正确的设备

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 如果需要，也可以加载优化器和调度器状态，以及历史记录
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if self.scheduler and 'scheduler_state_dict' in checkpoint:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.history = checkpoint.get('history', self.history) # 使用 get 以防旧检查点没有 history
        # self.best_val_acc = checkpoint.get('best_val_acc', self.best_val_acc)
        # start_epoch = checkpoint.get('epoch', 0)

        print(f"已从 {filepath} 加载模型权重。")
        return checkpoint  # 返回加载的检查点，可能包含其他信息如 epoch


# --------------------------------------------------------------------------------
# 主程序块 (用于演示或独立测试，实际使用时会在 run/*.py 中调用)
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # 这是一个非常简化的演示，实际使用时，你需要:
    # 1. 从 config.py 加载配置
    # 2. 创建 dataset 和 dataloaders (使用 dataset.py)
    # 3. 创建 model (使用 model.py)
    # 4. 定义 criterion, optimizer, scheduler
    # 5. 实例化 Trainer 并调用 .train()

    print("Trainer.py: 此脚本定义了 Trainer 类。")
    print("要运行实际训练，请使用 run/transfer.py 或 run/finetune.py。")

    # 示例：如何构建参数 (仅为结构示例，不运行)
    if False:  # 设置为 True 以查看结构，但不会运行
        # --- 1. 配置 ---
        device = config.DEVICE
        checkpoint_dir = config.CHECKPOINT_DIR
        epochs = 1  # 演示用

        # --- 2. 数据 ---
        # 假设 train_loader_demo, val_loader_demo 已通过 dataset.py 创建
        # train_loader_demo, val_loader_demo, _ = get_dataloaders(config.TRANSFER_LEARNING_BATCH_SIZE, config.TRANSFER_LEARNING_BATCH_SIZE)

        # --- 3. 模型 ---
        # model_demo = get_googlenet_model(
        #     num_classes=config.NUM_CLASSES,
        #     pretrained=True,
        #     freeze_extractor=True
        # ).to(device)

        # --- 4. 损失、优化器、调度器 ---
        # criterion_demo = nn.CrossEntropyLoss()
        # optimizer_demo = optim.Adam(
        #     filter(lambda p: p.requires_grad, model_demo.parameters()), # 只优化可训练参数
        #     lr=config.TRANSFER_LEARNING_LR,
        #     weight_decay=config.WEIGHT_DECAY
        # )
        # scheduler_demo = StepLR(optimizer_demo, step_size=config.STEP_LR_STEP_SIZE, gamma=config.STEP_LR_GAMMA)

        # --- 5. Trainer ---
        # trainer_demo = Trainer(
        #     model=model_demo,
        #     train_loader=train_loader_demo,
        #     val_loader=val_loader_demo,
        #     criterion=criterion_demo,
        #     optimizer=optimizer_demo,
        #     scheduler=scheduler_demo,
        #     device=device,
        #     epochs=epochs,
        #     checkpoint_dir=checkpoint_dir,
        #     model_name="demo_best_model.pth"
        # )
        # trainer_demo.train()
        pass