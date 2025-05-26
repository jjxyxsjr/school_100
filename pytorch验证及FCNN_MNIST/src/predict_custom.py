# src/predict_custom.py
import torch
import os
from PIL import Image as PIL_Image  # 用别名避免与我们自己的 Image 模块冲突 (如果有的话)
import matplotlib.pyplot as plt

from model import FCNN  # 你的模型类
from utils import preprocess_custom_image  # 我们刚创建的预处理函数
from data_loader import MNISTDataLoaders  # 用于获取 transform

# 中文显示配置 (可选，如果 utils.py 中的配置已全局生效)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败 (predict_custom.py)，某些标签可能无法正确显示: {e}")


def predict_single_image(model_path: str, image_path: str, data_dir_for_transform: str, device_str: str = 'cpu'):
    """
    加载模型并预测单个自定义图像。

    参数:
        model_path (str): 已训练模型的路径。
        image_path (str): 自定义图像的路径。
        data_dir_for_transform (str): 用于初始化 MNISTDataLoaders 以获取 transform 的数据目录。
        device_str (str): 'cuda' 或 'cpu'。
    """
    device = torch.device(device_str)

    # 1. 加载模型
    # 需要模型类定义来加载 state_dict
    model = FCNN().to(device)  # 使用 FCNN 的默认参数，确保它们与训练时一致
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 设置为评估模式
        print(f"模型已从 {model_path} 加载。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 获取用于 MNIST 的 transform
    # 我们需要从 data_loader 中获取与训练时完全一致的 transform
    # 这里我们实例化 MNISTDataLoaders 只是为了拿到它的 transform 属性
    # 注意：如果你的 FCNN 模型内部做了 Flatten，那么这里的 transform 不应该包含 Flatten。
    # 如果你的 FCNN 模型期望接收已经展平的输入，并且 Flatten 在 transform 中，那也是可以的。
    # 当前我们的 FCNN 模型内部有 nn.Flatten()，所以这里的 transform 不应包含 Flatten。
    # MNISTDataLoaders 构造函数中的 transform 也没有显式加入 Flatten。
    try:
        # 这里的 batch_size 和 shuffle 对获取 transform 不重要
        temp_data_loader_config = MNISTDataLoaders(data_dir=data_dir_for_transform, batch_size=1)
        mnist_transform = temp_data_loader_config.transform
        print("MNIST transform 获取成功。")
        # print(mnist_transform) # 可以打印出来看看包含哪些转换
    except Exception as e:
        print(f"获取 MNIST transform 失败: {e}")
        return

    # 3. 预处理自定义图像
    print(f"预处理图像: {image_path}")
    img_tensor = preprocess_custom_image(image_path, mnist_transform)

    if img_tensor is None:
        print("图像预处理失败。")
        return

    img_tensor = img_tensor.to(device)

    # 4. 进行预测
    with torch.no_grad():
        output = model(img_tensor)  # FCNN 模型内部会处理 Flatten
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_digit = predicted_class.item()
    prediction_confidence = confidence.item() * 100

    print(f"\n预测结果: {predicted_digit}")
    print(f"置信度: {prediction_confidence:.2f}%")

    # 5. 显示图像和预测结果
    try:
        original_img = PIL_Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)  # 如果是灰度图用 'gray'
        plt.title(f"自定义图像\n预测为: {predicted_digit} (置信度: {prediction_confidence:.2f}%)")
        plt.axis('off')

        # 保存图像 (可选)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_script_dir)
        results_dir = os.path.join(project_root_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "custom_image_prediction.png"))

        plt.show()  # 直接显示图像
        plt.close()
    except Exception as e:
        print(f"显示图像时出错: {e}")


if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)

    # !! 修改为你自己的图片路径 !!
    # 你的图片路径是 'pytorch验证及FCNN_MNIST/data/MNIST/processed/7.png'
    # 相对于项目根目录，它应该是 'data/MNIST/processed/7.png'
    CUSTOM_IMAGE_PATH = os.path.join(project_root_dir, "data", "MNIST", "processed", "7.png")

    MODEL_SAVE_PATH = os.path.join(project_root_dir, "saved_models", "mnist_fcnn_final.pth")

    # data_dir 用于获取与 MNIST 训练时相同的 transform
    DATA_DIR_FOR_TRANSFORM = os.path.join(project_root_dir, "data")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(CUSTOM_IMAGE_PATH):
        print(f"错误：自定义图像文件未找到于 {CUSTOM_IMAGE_PATH}")
    elif not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误：模型文件未找到于 {MODEL_SAVE_PATH}")
    else:
        predict_single_image(MODEL_SAVE_PATH, CUSTOM_IMAGE_PATH, DATA_DIR_FOR_TRANSFORM, DEVICE)