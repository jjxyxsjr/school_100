# src/evaluator.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report  # 用于更详细的指标
import seaborn as sns  # 用于绘制混淆矩阵
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np
import os

# --- 开始添加中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# --- 中文显示配置结束 ---
# 假设你的模型 (model.py) 和数据加载器 (data_loader.py) 在同一个 src 目录下
from model import FCNN  # 导入你的模型类


# from data_loader import MNISTDataLoaders # 如果你需要在这里实例化数据加载器

class ModelEvaluator:
    def __init__(self, model_class, model_path: str, device: str = 'cpu'):
        """
        初始化 ModelEvaluator。

        参数:
            model_class: 要加载的模型的类 (例如 FCNN)。
            model_path (str): 保存的模型 state_dict 的路径 (.pth 文件)。
            device (str): 运行评估的设备 ('cuda' 或 'cpu')。
        """
        self.device = device
        # 实例化模型架构。
        # 确保模型参数与保存的模型匹配 (例如 input_size, hidden_sizes, num_classes)
        self.model = model_class().to(self.device)  # 使用默认参数或从配置中加载

        # 加载保存的 state_dict
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型已成功从 {model_path} 加载")
        except Exception as e:
            print(f"从 {model_path} 加载模型时出错: {e}")
            raise

        self.model.eval()  # 将模型设置为评估模式

    @torch.no_grad()  # 装饰器，禁用梯度计算
    def visualize_predictions(self, data_loader: DataLoader, num_images: int = 10, save_path: str = None,
                              transform_for_display=None):
        """
        可视化一部分数据加载器中的图像及其预测结果。

        参数:
            data_loader (DataLoader): 包含测试/验证数据的数据加载器。
            num_images (int): 要显示的图像数量。
            save_path (str, optional): 可视化结果的保存路径。如果为 None，则不保存。
            transform_for_display (callable, optional): 用于反归一化图像以便显示的转换。
        """
        self.model.eval()
        images, labels = next(iter(data_loader))
        images_to_show = images[:num_images].to(self.device)
        labels_to_show = labels[:num_images].to(self.device)

        outputs = self.model(images_to_show)
        _, preds = torch.max(outputs, 1)

        images_to_show = images_to_show.cpu()  # 移回 CPU 以便绘图

        # 如果有反归一化变换，应用它
        # 注意：MNIST 的 Normalize 是 ((0.1307,), (0.3081,))
        # 反归一化：image = image * std + mean
        if transform_for_display is None:
            # 简单的反归一化 (假设原始数据是 0-1，然后被标准化)
            # 这个反向变换可能需要根据你的具体 transform 调整
            # 如果你的 transform 是 ToTensor() -> Normalize(mean, std)
            # 那么 images_to_show 已经是 [0,1] 区间然后标准化，这里直接显示可能偏暗或对比度不高
            # 为了更好的可视化，通常需要反归一化
            # 但简单的显示，即使不反归一化，数字的轮廓通常还是可见的。
            # 我们先直接显示，如果效果不好再实现精确的反归一化。
            pass

        fig = plt.figure(figsize=(15, 7 if num_images > 5 else 3))  # 调整图像大小
        cols = 5
        rows = (num_images + cols - 1) // cols  # 计算行数

        for idx in np.arange(num_images):
            ax = fig.add_subplot(rows, cols, idx + 1, xticks=[], yticks=[])
            # Matplotlib expects image shape (H, W, C) or (H, W) for grayscale
            # PyTorch tensor is (C, H, W). So we need to permute.
            # Also, remove channel dimension if it's 1 (grayscale)
            img_to_plot = images_to_show[idx].squeeze()  # (H, W) if C=1
            # 如果squeeze后还是3D (例如RGB)，则用 .permute(1, 2, 0)
            # if len(img_to_plot.shape) == 3 and img_to_plot.shape[0] in [1,3]: # C, H, W
            #    img_to_plot = img_to_plot.permute(1, 2, 0).squeeze()

            ax.imshow(img_to_plot, cmap='gray')  # MNIST 是灰度图
            title_color = "green" if preds[idx] == labels_to_show[idx] else "red"
            ax.set_title(f"预测: {preds[idx].item()}\n(真实: {labels_to_show[idx].item()})", color=title_color)

        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"预测可视化图像已保存到: {save_path}")
            except Exception as e:
                print(f"保存预测可视化图像失败: {e}")
        # plt.show()
        plt.close()
    def evaluate(self, test_loader: DataLoader, results_dir: str = None):
        """
        在给定的 test_loader 上评估模型。

        参数:
            test_loader (DataLoader): 测试集的数据加载器。
            results_dir (str, optional): 保存评估结果 (如混淆矩阵) 的目录。

        返回:
            dict: 包含评估指标 (例如准确率、报告) 的字典。
        """
        all_preds = []
        all_labels = []
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()  # 如果需要，用于计算损失

        print(f"\n在 {self.device} 上进行评估...")
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * total_correct / total_samples
        avg_loss = running_loss / total_samples

        print(f"测试准确率 (Test Accuracy): {accuracy:.2f}%")
        print(f"平均测试损失 (Average Test Loss): {avg_loss:.4f}")

        # 分类报告 (精确率、召回率、F1 分数)
        class_names = [str(i) for i in range(10)]  # MNIST 类别名称 (0-9)
        report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        print("\n分类报告 (Classification Report):")
        print(report_str)

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print("\n混淆矩阵 (Confusion Matrix):")
        print(cm)

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            # 绘制并保存混淆矩阵
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('预测标签 (Predicted Label)')
            plt.ylabel('真实标签 (True Label)')
            plt.title('混淆矩阵 (Confusion Matrix)')
            cm_path = os.path.join(results_dir, "confusion_matrix.png")
            try:
                plt.savefig(cm_path)
                print(f"混淆矩阵图已保存到 {cm_path}")
            except Exception as e:
                print(f"保存混淆矩阵图时出错: {e}")
            plt.close()

            # 将分类报告保存到文件
            report_path = os.path.join(results_dir, "classification_report.txt")
            try:
                with open(report_path, "w", encoding="utf-8") as f:  # 添加 encoding="utf-8"
                    f.write("测试准确率 (Test Accuracy): {:.2f}%\n".format(accuracy))
                    f.write("平均测试损失 (Average Test Loss): {:.4f}\n\n".format(avg_loss))
                    f.write(report_str)
                print(f"分类报告已保存到 {report_path}")
            except Exception as e:
                print(f"保存分类报告时出错: {e}")

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "classification_report": report_str,
            "confusion_matrix": cm
        }


if __name__ == '__main__':
    # 这个代码块允许独立执行评估器
    # 你需要在这里配置路径和数据加载。

    # --- 独立评估配置 ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)

    MODEL_PATH = os.path.join(project_root_dir, "saved_models", "mnist_fcnn_final.pth")
    DATA_DIR = os.path.join(project_root_dir, "data")
    RESULTS_DIR = os.path.join(project_root_dir, "results")  # 用于保存图表/报告
    BATCH_SIZE = 128  # 或任何适合评估的批处理大小
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 在 {MODEL_PATH} 未找到模型文件")
        print("请确保你已经训练了模型并且它已正确保存。")
    else:
        print(f"尝试评估模型: {MODEL_PATH}")
        print(f"使用设备: {DEVICE}")

        # 1. 加载测试数据
        # 你可能需要导入并使用你的 MNISTDataLoaders 类
        from data_loader import MNISTDataLoaders  # 确保这个导入能正常工作

        try:
            data_loaders = MNISTDataLoaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
            test_loader = data_loaders.get_test_loader()
            print("测试数据加载成功。")
        except Exception as e:
            print(f"加载测试数据时出错: {e}")
            test_loader = None  # 如果加载失败，确保 test_loader 为 None

        if test_loader:
            # 2. 初始化评估器并进行评估
            try:
                # 传递 FCNN 类本身，而不是一个实例
                evaluator = ModelEvaluator(model_class=FCNN, model_path=MODEL_PATH, device=DEVICE)
                evaluation_results = evaluator.evaluate(test_loader, results_dir=RESULTS_DIR)
                print("\n评估完成。")
            except Exception as e:
                print(f"评估过程中发生错误: {e}")
        if test_loader:
            try:
                evaluator = ModelEvaluator(model_class=FCNN, model_path=MODEL_PATH, device=DEVICE)
                evaluation_results = evaluator.evaluate(test_loader, results_dir=RESULTS_DIR)
                print("\n评估完成。")

                # 新增：调用可视化函数
                vis_save_path = os.path.join(RESULTS_DIR, "test_predictions_visualization.png")
                evaluator.visualize_predictions(test_loader, num_images=10, save_path=vis_save_path)

            except Exception as e:
                print(f"评估或可视化过程中发生错误: {e}")