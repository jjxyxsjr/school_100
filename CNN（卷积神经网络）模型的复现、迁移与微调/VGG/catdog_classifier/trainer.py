import os
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from dataset import CatDogDataset  # 确保 dataset.py 文件存在且路径正确
from model import build_vgg16_model  # 确保 model.py 文件存在且路径正确


class CatDogTrainer:
    def __init__(self, train_dir, test_dir, transform, skip_train=False):
        """
        初始化训练器。
        参数:
            train_dir (str): 训练数据目录路径。
            test_dir (str): 测试数据目录路径。
            transform (callable): 应用于图像的预处理转换。
            skip_train (bool): 如果为True且模型权重存在，则跳过训练。
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform
        self.skip_train = skip_train

        self.batch_size = 20  # 训练和测试时的批处理大小
        self.epochs = 10  # 训练的总轮数

        # 根据可用性选择设备 (GPU 或 CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 构建VGG16模型并将其移动到选定的设备
        self.model = build_vgg16_model().to(self.device)
        # 定义模型检查点（权重）的保存路径
        self.checkpoint_path = "./checkpoints/catdog_vgg16.pth"
        # 定义优化器，这里使用带动量的随机梯度下降 (SGD)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # 定义损失函数，这里使用交叉熵损失，适用于多分类问题
        self.loss_fn = nn.CrossEntropyLoss()

        # 初始化列表，用于存储每个epoch的训练损失和测试损失，以便后续绘图
        self.train_losses = []
        self.test_losses = []
        # (可选) 初始化列表，用于存储每个epoch的训练准确率和测试准确率
        # self.train_accuracies = []
        # self.test_accuracies = []

    def load_data(self):
        """加载训练和测试数据集。"""
        # 创建训练数据集实例
        train_dataset = CatDogDataset(self.train_dir, self.transform)
        # 创建测试数据集实例
        test_dataset = CatDogDataset(self.test_dir, self.transform)
        # 创建训练数据加载器，用于批量加载数据，并在每个epoch开始时打乱顺序
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 创建测试数据加载器
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        print(f"数据加载完毕：训练集图像 {len(train_dataset)} 张，测试集图像 {len(test_dataset)} 张。")

    def save_model(self):
        """保存当前模型的权重到检查点文件。"""
        # 确保保存路径的目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        # 保存模型的状态字典 (包含所有可学习的参数)
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"✅ 模型已保存至 {self.checkpoint_path}")

    def load_model(self):
        """从检查点文件加载模型权重。"""
        if os.path.exists(self.checkpoint_path):  # 检查权重文件是否存在
            try:
                # 加载权重到模型，map_location确保权重能正确加载到当前设备
                self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
                self.model.eval()  # 加载后将模型设置为评估模式，这会关闭dropout等层
                print(f"✅ 已加载模型权重：{self.checkpoint_path}")
                return True  # 返回True表示加载成功
            except Exception as e:
                print(f"⚠️ 加载模型权重失败：{e}。将重新训练。")
                return False  # 返回False表示加载失败
        else:
            print("ℹ️ 未找到已保存模型，将重新训练。")
            return False  # 未找到文件，返回False

    def evaluate(self, loader, label="测试"):
        """
        在给定的数据加载器上评估模型。
        参数:
            loader (DataLoader): 用于评估的数据加载器 (可以是训练或测试加载器)。
            label (str): 评估的标签 (例如 "测试", "训练")，用于打印信息。
        返回:
            tuple: 包含平均损失 (float) 和准确率 (float) 的元组。
        """
        self.model.eval()  # 确保模型处于评估模式
        correct = 0  # 正确预测的数量
        total_loss = 0  # 当前数据集上的总损失
        dataset_size = len(loader.dataset)  # 数据集总大小

        with torch.no_grad():  # 在评估期间不计算梯度，以节省内存和计算
            for x, y in loader:  # 遍历数据加载器中的每个批次
                x, y = x.to(self.device), y.to(self.device)  # 将数据移动到设备
                pred = self.model(x)  # 模型前向传播，得到预测结果
                loss = self.loss_fn(pred, y)  # 计算损失
                total_loss += loss.item() * x.size(0)  # 累加批次损失 (乘以批次大小以得到批次总损失)

                pred_labels = torch.argmax(pred, dim=1)  # 获取预测概率最高的类别作为预测标签
                correct += (pred_labels == y).sum().item()  # 统计正确预测的数量

        avg_loss = total_loss / dataset_size  # 计算平均损失
        acc = correct / dataset_size  # 计算准确率

        print(f"{label}集准确率: {acc * 100:.2f}% | 平均损失: {avg_loss:.4f}")
        return avg_loss, acc  # 返回平均损失和准确率

    def train(self):
        """执行模型的训练过程。"""
        print(f"开始在 {self.device} 上进行训练，共 {self.epochs} 个 epochs...")
        start_time = time.time()  # 记录训练开始时间

        for epoch in range(self.epochs):  # 遍历每个epoch
            self.model.train()  # 在每个epoch的训练阶段开始时，确保模型处于训练模式

            epoch_train_loss_sum = 0.0  # 当前epoch的训练总损失累加器
            # epoch_train_correct = 0 # (可选) 当前epoch的训练正确数累加器

            # 遍历训练数据加载器中的每个批次
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)  # 将数据移动到设备

                # 前向传播
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                # 反向传播和优化
                self.optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 计算当前批次的梯度
                self.optimizer.step()  # 根据梯度更新模型参数

                epoch_train_loss_sum += loss.item() * x.size(0)  # 累加批次损失
                # pred_labels_train = torch.argmax(pred, dim=1) # (可选) 获取训练批次的预测标签
                # epoch_train_correct += (pred_labels_train == y).sum().item() # (可选) 累加训练批次的正确数

                # 每处理50个批次打印一次当前批次的损失，方便监控训练进度
                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} | Batch {batch_idx}/{len(self.train_loader)} | Batch Loss: {loss.item():.4f}")

            # 计算当前epoch的平均训练损失
            avg_epoch_train_loss = epoch_train_loss_sum / len(self.train_loader.dataset)
            self.train_losses.append(avg_epoch_train_loss)  # 记录到列表中
            # avg_epoch_train_acc = epoch_train_correct / len(self.train_loader.dataset) # (可选) 计算平均训练准确率
            # self.train_accuracies.append(avg_epoch_train_acc) # (可选) 记录

            # 在每个epoch训练完成后，在测试集上评估模型性能
            # evaluate 方法内部会将模型设置为 self.model.eval()
            avg_epoch_test_loss, avg_epoch_test_acc = self.evaluate(self.test_loader, f"测试 (Epoch {epoch + 1})")
            self.test_losses.append(avg_epoch_test_loss)  # 记录当前epoch的测试损失
            # self.test_accuracies.append(avg_epoch_test_acc) # (可选) 记录当前epoch的测试准确率

            # 打印当前epoch的训练和测试性能总结
            print(f"--- Epoch {epoch + 1}/{self.epochs} 总结 ---")
            print(f"平均训练损失: {avg_epoch_train_loss:.4f}")  # | (可选) 训练准确率: {avg_epoch_train_acc*100:.2f}%
            print(f"平均测试损失: {avg_epoch_test_loss:.4f} | 测试准确率: {avg_epoch_test_acc * 100:.2f}%")
            print("-" * 30)  # 分隔符

        training_duration = time.time() - start_time  # 计算总训练时长
        print(f"训练完成，总耗时: {training_duration:.1f} 秒 ({training_duration / 60:.2f} 分钟)")
        self.save_model()  # 训练完成后保存模型

    def plot_losses(self):
        """绘制每个epoch的训练损失和测试损失曲线图。"""
        plt.figure(figsize=(12, 6))  # 设置图像大小，使其更易读

        # 绘制训练损失曲线
        if self.train_losses:
            plt.plot(self.train_losses, label="Train Loss per Epoch", marker='o', linestyle='-')
        # 绘制测试损失曲线
        if self.test_losses:
            plt.plot(self.test_losses, label="Test Loss per Epoch", marker='x', linestyle='--')

        # 设置X轴刻度标签为Epoch编号 (1, 2, ..., N)
        if self.epochs > 0:
            tick_labels = [str(i + 1) for i in range(self.epochs)]  # 生成Epoch标签列表
            # 如果Epoch数量不多 (例如<=20)，则显示所有Epoch刻度
            if self.epochs <= 20:
                plt.xticks(range(self.epochs), tick_labels)
            else:  # 如果Epoch数量较多，则每隔一定步长显示一个刻度，避免拥挤
                step = max(1, self.epochs // 10)  # 计算步长，确保至少为1，大约显示10个刻度
                plt.xticks(range(0, self.epochs, step), [tick_labels[i] for i in range(0, self.epochs, step)])

        plt.xlabel("Epoch")  # X轴标签
        plt.ylabel("Loss")  # Y轴标签
        plt.title("Train vs Test Loss per Epoch")  # 图像标题
        plt.grid(True)  # 添加网格线，方便查看数值

        # 获取图例句柄和标签，仅在有可绘制的线条时显示图例
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()  # 显示图例

        plot_filename = "epoch_loss_curves.png"  # 定义保存图像的文件名
        try:
            plt.savefig(plot_filename)  # 保存图像
            print(f"📈 损失曲线图已保存至 {plot_filename}")
        except Exception as e:
            print(f"⚠️ 保存损失曲线图失败: {e}")
        finally:
            plt.close('all')  # 关闭所有matplotlib图像窗口，释放资源

    def run(self):
        """执行训练器的主流程：加载数据、训练 (或跳过)、评估、绘图。"""
        self.load_data()  # 首先加载数据

        # 根据 self.skip_train 决定是否跳过训练
        if self.skip_train:
            loaded_successfully = self.load_model()  # 尝试加载已保存的模型
            if loaded_successfully:
                print("已加载预训练模型，跳过训练，直接进行最终评估。")
                # 如果模型加载成功，直接在测试集上进行一次最终评估
                self.evaluate(self.test_loader, "最终测试评估 (加载模型后)")
                return  # 结束run方法
            else:
                # 如果skip_train为True但模型加载失败 (例如文件不存在或损坏)
                print("警告：设置了skip_train=True，但无法加载模型或模型不存在。将开始新的训练。")

        # 如果不跳过训练，或者跳过训练但模型加载失败，则执行训练流程
        self.train()

        # 训练完成后，在训练集和测试集上进行最终的评估并打印结果
        print("\n--- 训练完成后的最终评估 ---")
        self.evaluate(self.train_loader, "最终训练集评估")
        self.evaluate(self.test_loader, "最终测试集评估")

        # 如果执行了训练 (即train_losses列表不为空)，则绘制并保存损失曲线图
        if self.train_losses and self.test_losses:
            self.plot_losses()
        else:
            # 如果没有训练数据 (例如，直接加载模型并评估，或者epochs为0)
            print("ℹ️ 未执行训练或损失数据不足，不生成损失曲线图。")