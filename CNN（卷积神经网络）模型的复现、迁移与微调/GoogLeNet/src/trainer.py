# src/trainer.py
# 管理训练和验证逻辑，包括提前停止和日志记录

import torch  # 用于张量操作和设备管理
import torch.nn as nn  # 提供神经网络组件（如损失函数）
import torch.optim as optim  # 提供优化器（如 Adam）
import time  # 用于记录训练时间
import copy  # 用于深拷贝模型权重
import csv  # 用于写入 CSV 日志
import os  # 用于文件操作

class Trainer:
    def __init__(self, config, model, dataloaders, dataset_sizes, device):
        """初始化训练器，设置模型、数据加载器和设备"""
        self.config = config  # 存储配置对象，包含超参数
        self.model = model  # 存储 Inception V3 模型
        self.dataloaders = dataloaders  # 存储训练和验证数据加载器
        self.dataset_sizes = dataset_sizes  # 存储数据集大小
        self.device = device  # 存储计算设备（GPU 或 CPU）
        self.criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    def train(self, num_epochs, learning_rate, phase_name="训练", current_best_acc=0.0, patience=5):
        """执行训练和验证，包含提前停止和日志记录"""
        since = time.time()  # 记录训练开始时间
        val_acc_history = []  # 存储验证准确率历史
        train_acc_history = []  # 存储训练准确率历史
        val_loss_history = []  # 存储验证损失历史
        train_loss_history = []  # 存储训练损失历史

        best_model_wts = copy.deepcopy(self.model.state_dict())  # 深拷贝初始模型权重，用于保存最佳模型
        best_acc = current_best_acc  # 初始化最佳验证准确率
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],  # 仅优化可训练参数
            lr=learning_rate,  # 设置学习率（如初始训练 1e-3，微调 1e-5）
            weight_decay=self.config.OPTIMIZED_WEIGHT_DECAY  # 设置权重衰减（1e-4），正则化
        )

        # 初始化 CSV 日志
        write_header = not os.path.exists(self.config.CSV_LOG_PATH) or os.path.getsize(self.config.CSV_LOG_PATH) == 0  # 检查是否需要写入表头
        with open(self.config.CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as csv_file:  # 打开 CSV 文件，追加模式
            csv_writer = csv.writer(csv_file)  # 创建 CSV 写入器
            if write_header:
                csv_writer.writerow(['epoch', 'phase', 'loss', 'accuracy', 'timestamp'])  # 写入表头：轮次、阶段、损失、准确率、时间戳

            epochs_no_improve = 0  # 记录验证准确率未提升的轮次，用于提前停止
            stopped_epoch = num_epochs - 1  # 初始化停止轮次，默认为最后轮次

            for epoch in range(num_epochs):  # 遍历指定轮次
                print(f'\n轮次 {epoch + 1}/{num_epochs} ({phase_name})')  # 打印当前轮次和阶段（如“训练”或“微调”）
                print('-' * 10)  # 分隔线，提升可读性

                for phase in ['train', 'validation']:  # 遍历训练和验证阶段
                    phase_cn = "训练" if phase == 'train' else "验证"  # 设置中文阶段名称
                    if phase == 'train':
                        self.model.train()  # 设置模型为训练模式（启用 Dropout 和 BatchNorm）
                    else:
                        self.model.eval()  # 设置模型为评估模式（禁用 Dropout 和 BatchNorm）

                    running_loss = 0.0  # 初始化当前阶段的累计损失
                    running_corrects = 0  # 初始化当前阶段的正确预测数

                    for inputs, labels in self.dataloaders[phase]:  # 遍历数据加载器，获取批量数据
                        inputs = inputs.to(self.device)  # 将输入图像移动到指定设备（GPU 或 CPU）
                        labels = labels.to(self.device)  # 将标签移动到指定设备
                        optimizer.zero_grad()  # 清空优化器的梯度，准备新一轮计算

                        with torch.set_grad_enabled(phase == 'train'):  # 控制梯度计算（训练时启用，验证时禁用）
                            # 处理 Inception V3 的辅助输出
                            if phase == 'train' and self.model.aux_logits and hasattr(self.model, 'AuxLogits') and self.model.AuxLogits is not None:
                                # 训练阶段：Inception V3 返回主输出和辅助输出
                                outputs, aux_outputs = self.model(inputs)  # 获取主分类器和辅助分类器输出
                                loss1 = self.criterion(outputs, labels)  # 计算主分类器损失（交叉熵）
                                loss2 = self.criterion(aux_outputs, labels)  # 计算辅助分类器损失
                                loss = loss1 + 0.4 * loss2  # 总损失为主损失加 0.4 倍辅助损失，增强梯度流动
                            else:
                                # 验证阶段或无辅助输出：仅使用主分类器输出
                                outputs = self.model(inputs)  # 获取主分类器输出
                                loss = self.criterion(outputs, labels)  # 计算交叉熵损失

                            _, preds = torch.max(outputs, 1)  # 获取预测类别（最大概率的索引）

                            if phase == 'train':
                                loss.backward()  # 反向传播，计算梯度
                                optimizer.step()  # 更新模型参数

                        running_loss += loss.item() * inputs.size(0)  # 累加批量损失（乘以批量大小）
                        running_corrects += torch.sum(preds == labels.data)  # 累加正确预测数

                    epoch_loss = running_loss / self.dataset_sizes[phase]  # 计算平均损失
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]  # 计算平均准确率

                    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())  # 获取当前时间戳
                    print(f'{phase_cn}损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')  # 打印损失和准确率
                    csv_writer.writerow([epoch, phase, f"{epoch_loss:.4f}", f"{epoch_acc:.4f}", current_time_str])  # 写入 CSV 日志
                    csv_file.flush()  # 刷新文件，确保日志实时写入

                    if phase == 'validation':
                        val_loss_history.append(epoch_loss)  # 记录验证损失
                        val_acc_history.append(epoch_acc.item())  # 记录验证准确率
                        if epoch_acc > best_acc:  # 检查是否为最佳验证准确率
                            best_acc = epoch_acc  # 更新最佳准确率
                            best_model_wts = copy.deepcopy(self.model.state_dict())  # 保存最佳模型权重
                            torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)  # 保存模型到指定路径
                            print(f"最佳验证准确率提升至 {best_acc:.4f}。模型已保存至 {self.config.MODEL_SAVE_PATH}")
                            epochs_no_improve = 0  # 重置未提升计数器
                        else:
                            epochs_no_improve += 1  # 增加未提升计数
                    else:
                        train_loss_history.append(epoch_loss)  # 记录训练损失
                        train_acc_history.append(epoch_acc.item())  # 记录训练准确率

                if epochs_no_improve >= patience:  # 检查是否触发提前停止
                    print(f'提前终止于第 {epoch + 1} 轮。')  # 打印提前终止信息
                    stopped_epoch = epoch  # 记录停止轮次
                    self.model.load_state_dict(best_model_wts)  # 恢复最佳模型权重
                    return self.model, train_acc_history, val_acc_history, train_loss_history, val_loss_history, best_acc, stopped_epoch  # 返回结果

        time_elapsed = time.time() - since  # 计算训练总耗时
        print(f'{phase_name}阶段在 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒 内完成')  # 打印耗时
        print(f'最佳验证准确率: {best_acc:4f}')  # 打印最佳验证准确率

        self.model.load_state_dict(best_model_wts)  # 恢复最佳模型权重
        return self.model, train_acc_history, val_acc_history, train_loss_history, val_loss_history, best_acc, stopped_epoch  # 返回最终结果