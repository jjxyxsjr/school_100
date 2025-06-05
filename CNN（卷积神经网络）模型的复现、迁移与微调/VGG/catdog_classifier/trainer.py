import os
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from dataset import CatDogDataset
# from model import build_vgg16_model, unfreeze_last_conv_block
from model import build_vgg16_model, unfreeze_all_layers


class CatDogTrainer:
    def __init__(self, train_dir, test_dir, transform, epochs=10, fine_tune_epochs=10, fine_tune_lr=0.0001):
        """
        初始化训练器。
        参数:
            train_dir (str): 训练数据目录。
            test_dir (str): 测试数据目录。
            transform (callable): 图像预处理。
            epochs (int): 第一阶段（迁移学习）的训练轮数。
            fine_tune_epochs (int): 第二阶段（微调）的训练轮数。
            fine_tune_lr (float): 微调阶段的学习率。
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr = fine_tune_lr

        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build_vgg16_model() 不再需要参数
        self.model = build_vgg16_model().to(self.device)

        self.checkpoint_path = "checkpoints/best_catdog_vgg16.pth"
        self.loss_fn = nn.CrossEntropyLoss()

        # 历史记录
        self.history = {
            'stage1': {'train_loss': [], 'test_loss': [], 'test_acc': []},
            'stage2': {'train_loss': [], 'test_loss': [], 'test_acc': []}
        }
        self.best_acc = 0.0  # 用于追踪最佳准确率

    def load_data(self):
        """加载训练和测试数据集。"""
        train_dataset = CatDogDataset(self.train_dir, self.transform)
        test_dataset = CatDogDataset(self.test_dir, self.transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        print(f"数据加载完毕：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张。")

    def save_model(self, current_acc):
        """仅当模型性能提升时，保存最佳模型。"""
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"✅ 新的最佳模型已保存！准确率: {current_acc * 100:.2f}%")

    def evaluate(self, loader):
        """在给定的数据加载器上评估模型，返回平均损失和准确率。"""
        self.model.eval()
        correct = 0
        total_loss = 0
        dataset_size = len(loader.dataset)

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item() * x.size(0)
                pred_labels = torch.argmax(pred, dim=1)
                correct += (pred_labels == y).sum().item()

        avg_loss = total_loss / dataset_size
        acc = correct / dataset_size
        return avg_loss, acc

    def _train_one_epoch(self, epoch, total_epochs, optimizer):
        """执行一个 epoch 的训练逻辑。"""
        self.model.train()
        epoch_train_loss_sum = 0.0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss_sum += loss.item() * x.size(0)
            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch}/{total_epochs} | Batch {batch_idx}/{len(self.train_loader)} | Batch Loss: {loss.item():.4f}")

        return epoch_train_loss_sum / len(self.train_loader.dataset)

    def train(self):
        """执行完整的两阶段训练：迁移学习 + 微调。"""
        print(f"设备: {self.device}")
        start_time = time.time()

        # --- 阶段一：迁移学习（只训练分类头） ---
        print("\n--- 开始第一阶段：迁移学习 (Feature Extraction) ---")
        params_to_update_stage1 = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer_stage1 = torch.optim.SGD(params_to_update_stage1, lr=0.01, momentum=0.9)
        # 在 CatDogTrainer 类的 train 方法中
        optimizer_stage1 = torch.optim.SGD(params_to_update_stage1, lr=0.001, momentum=0.9)

        for epoch in range(1, self.epochs + 1):
            avg_train_loss = self._train_one_epoch(epoch, self.epochs, optimizer_stage1)
            avg_test_loss, avg_test_acc = self.evaluate(self.test_loader)

            print(f"--- [阶段一] Epoch {epoch}/{self.epochs} 总结 ---")
            print(f"训练损失: {avg_train_loss:.4f} | 测试损失: {avg_test_loss:.4f} | 测试准确率: {avg_test_acc * 100:.2f}%")

            self.history['stage1']['train_loss'].append(avg_train_loss)
            self.history['stage1']['test_loss'].append(avg_test_loss)
            self.history['stage1']['test_acc'].append(avg_test_acc)
            self.save_model(avg_test_acc)

        # --- 阶段二：微调（仅解冻最后的卷积块和分类器） ---
        print(f"\n--- 开始第二阶段：微调 (Fine-tuning last block) ---")
        # unfreeze_last_conv_block(self.model)
        unfreeze_all_layers(self.model)
        params_to_update_stage2 = [p for p in self.model.parameters() if p.requires_grad]
        print(f"\n微调阶段将更新 {len(params_to_update_stage2)} 个参数张量。")
        optimizer_stage2 = torch.optim.SGD(params_to_update_stage2, lr=self.fine_tune_lr, momentum=0.9)

        total_epochs_combined = self.epochs + self.fine_tune_epochs
        for epoch in range(self.epochs + 1, total_epochs_combined + 1):
            current_fine_tune_epoch = epoch - self.epochs
            avg_train_loss = self._train_one_epoch(f"{current_fine_tune_epoch}/{self.fine_tune_epochs}", "微调",
                                                 optimizer_stage2)
            avg_test_loss, avg_test_acc = self.evaluate(self.test_loader)

            print(f"--- [阶段二] Epoch {current_fine_tune_epoch}/{self.fine_tune_epochs} 总结 ---")
            print(f"训练损失: {avg_train_loss:.4f} | 测试损失: {avg_test_loss:.4f} | 测试准确率: {avg_test_acc * 100:.2f}%")

            self.history['stage2']['train_loss'].append(avg_train_loss)
            self.history['stage2']['test_loss'].append(avg_test_loss)
            self.history['stage2']['test_acc'].append(avg_test_acc)
            self.save_model(avg_test_acc)

        training_duration = time.time() - start_time
        print(f"\n训练完成，总耗时: {training_duration:.1f} 秒 ({training_duration / 60:.2f} 分钟)")
        print(f"🏆 最高测试准确率: {self.best_acc * 100:.2f}%，模型已保存在 {self.checkpoint_path}")

    def plot_curves(self):
        """绘制曲线图，使用全局设置显示中文。"""
        # 用两行代码设置全局中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 指定中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        plt.figure(figsize=(18, 6))

        epochs_stage1 = range(1, self.epochs + 1)
        epochs_stage2 = range(self.epochs + 1, self.epochs + self.fine_tune_epochs + 1)

        # 1. 损失曲线
        plt.subplot(1, 2, 1)
        plt.title('训练与测试损失曲线')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.plot(epochs_stage1, self.history['stage1']['train_loss'], 'bo-', label='阶段1 - 训练损失')
        plt.plot(epochs_stage1, self.history['stage1']['test_loss'], 'ro-', label='阶段1 - 测试损失')
        if self.fine_tune_epochs > 0:
            plt.plot(epochs_stage2, self.history['stage2']['train_loss'], 'g*--', label='阶段2 - 训练损失')
            plt.plot(epochs_stage2, self.history['stage2']['test_loss'], 'm*--', label='阶段2 - 测试损失')
            plt.axvline(x=self.epochs + 0.5, color='gray', linestyle='--', label='阶段分割线')

        plt.legend()
        plt.grid(True)

        # 2. 准确率曲线
        plt.subplot(1, 2, 2)
        plt.title('测试准确率曲线')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.plot(epochs_stage1, self.history['stage1']['test_acc'], 'ro-', label='阶段1 - 测试准确率')
        if self.fine_tune_epochs > 0:
            plt.plot(epochs_stage2, self.history['stage2']['test_acc'], 'm*--', label='阶段2 - 测试准确率')
            plt.axvline(x=self.epochs + 0.5, color='gray', linestyle='--', label='阶段分割线')

        plt.legend()
        plt.grid(True)

        plot_filename = "training_curves_fine_tuned.png"
        plt.savefig(plot_filename)
        print(f"📈 训练曲线图已保存至 {plot_filename}")
        plt.close('all')

    def run(self):
        """执行训练器的主流程。"""
        self.load_data()
        self.train()
        self.plot_curves()