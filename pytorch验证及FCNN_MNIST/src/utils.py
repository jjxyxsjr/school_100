# src/utils.py
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torchvision import transforms
# 中文显示配置 (确保在所有绘图前配置好)
# 你可以选择一个你系统上安装好的支持中文的字体
# 常见的选择有 'SimHei' (黑体), 'Microsoft YaHei' (微软雅黑), 'Source Han Sans CN' (思源黑体)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"设置中文字体失败，某些标签可能无法正确显示: {e}")
    print("请确保你选择的字体已安装在系统中。常用的中文字体有 'SimHei', 'Microsoft YaHei' 等。")


def plot_training_history(history: dict, save_path: str = None):
    """
    绘制训练过程中的损失和准确率曲线。

    参数:
        history (dict): 包含 'train_losses', 'val_losses', 'train_accuracies',
                        'val_accuracies' 列表的字典。
        save_path (str, optional): 图表保存路径。如果为 None，则不保存。
    """
    epochs = range(1, len(history['train_losses']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_losses'], 'bo-', label='训练损失 (Train Loss)')
    if history.get('val_losses') and len(history['val_losses']) > 0:  # 检查是否有验证损失
        plt.plot(epochs, history['val_losses'], 'ro-', label='验证损失 (Val Loss)')
    plt.title('训练和验证损失曲线 (Training and Validation Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracies'], 'bo-', label='训练准确率 (Train Acc)')
    if history.get('val_accuracies') and len(history['val_accuracies']) > 0:  # 检查是否有验证准确率
        plt.plot(epochs, history['val_accuracies'], 'ro-', label='验证准确率 (Val Acc)')
    plt.title('训练和验证准确率曲线 (Training and Validation Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('准确率 (Accuracy) (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 调整子图布局

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"训练历史图表已保存到: {save_path}")
        except Exception as e:
            print(f"保存训练历史图表失败: {e}")

    # 如果你希望在运行时直接显示图表 (例如在 Jupyter Notebook 或某些 IDE 中)
    # 可以取消下面这行的注释，但在服务器或无 GUI 环境下运行时要小心。
    # plt.show()
    plt.close()  # 保存后关闭图像，避免在内存中累积


def preprocess_custom_image(image_path: str, target_transform: transforms.Compose) -> torch.Tensor:
    """
    加载自定义手写数字图像，并进行预处理以匹配模型的输入。

    参数:
        image_path (str): 自定义图像的路径。
        target_transform (transforms.Compose): 用于 MNIST 数据的转换序列
                                               (应包含 ToTensor, Normalize)。
                                               注意：如果原始的 transform 包含 Flatten，
                                               这里应该能处理，或者模型本身有内部的 Flatten 层。

    返回:
        torch.Tensor: 预处理后的图像张量，准备好输入模型 (通常形状为 [1, C, H, W] 或 [1, Features])。
                      如果发生错误则返回 None。
    """
    try:
        # 1. 打开图像并转换为灰度图 ('L'模式)
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"错误：在路径 {image_path} 未找到图像文件。")
        return None
    except Exception as e:
        print(f"打开或转换图像时出错：{e}")
        return None

    # 2. 调整大小到 28x28 (MNIST 图像的标准大小)
    # 你可以根据需要选择不同的插值方法，如 Image.LANCZOS, Image.BILINEAR, Image.NEAREST
    img = img.resize((28, 28), Image.LANCZOS)

    # 3. 颜色反转 (对于大多数自定义手写图片，这一步非常关键!)
    # MNIST 数据集是白字黑底 (即数字部分的像素值较高，背景部分的像素值较低)。
    # 如果你的手写数字是像在白纸上用黑笔写的那样 (黑字白底)，
    # 那么你需要反转图像颜色，使得数字变白，背景变黑。
    #
    # 假设你的 "7.png" 是黑字白底，那么下面的反转是必要的。
    # 如果你的 "7.png" 已经是白字黑底 (像扫描的 MNIST 样本)，则可以注释掉或移除反转步骤。
    #
    # 使用 Pillow 的 ImageOps 进行反转:
    try:
        img = ImageOps.invert(img)
        print(f"图像 {image_path} 已执行颜色反转 (假设为黑字白底输入)。")
    except Exception as e:
        print(f"颜色反转时出错：{e}")
        # 可以选择是否在此处返回 None，或者继续尝试不反转的版本
        # return None

    # (或者，如果你想手动控制像素值进行反转，可以这样做):
    # np_img = np.array(img)
    # np_img = 255 - np_img # 假设是 8-bit 图像
    # img = Image.fromarray(np_img)

    # 4. 应用与训练 MNIST 数据时相同的转换
    # target_transform 通常包含 ToTensor() (将图像转换为 PyTorch 张量并使像素值在 [0,1] 区间)
    # 和 Normalize() (根据 MNIST 的均值和标准差进行归一化)。
    # 注意：我们的 FCNN 模型内部有 nn.Flatten() 层，所以这里的 transform 不需要包含 Flatten。
    try:
        tensor_img = target_transform(img)
    except Exception as e:
        print(f"应用 transform 时出错：{e}")
        return None

    # 5. 增加一个批次维度 (模型期望输入形状为 [N, C, H, W] 或 [N, Features])
    # 对于单个图像，N=1。
    tensor_img = tensor_img.unsqueeze(0)

    return tensor_img