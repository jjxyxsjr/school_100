# 数据集: Hateful Memes (我们将只下载少量样本用于本地快速演示)
# 模型: OpenAI 的 CLIP (ViT-B/32 版本)
# 任务: 判断一张图文结合的 Meme 是否是仇恨言论 (二分类)
# 实现: Python 脚本 (.py)，使用 transformers, torch, Pillow, matplotlib 库。
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import os

# --- 1. 加载预训练模型和处理器 ---
# 模型名称
MODEL_NAME = "openai/clip-vit-base-patch32"

# 打印提示信息
print(f"正在加载模型: {MODEL_NAME}...")
print("第一次运行时，会自动下载模型文件 (约 604 MB)，请耐心等待。")

# 加载模型和对应的预处理器
# CLIPModel是模型主体，CLIPProcessor负责将文本和图像转换成模型能理解的格式
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("模型加载完毕！")


# --- 2. 定义分类函数 ---
def classify_meme(image_path, text_labels):
    """
    使用 CLIP 模型对给定的图像和文本标签进行分类。

    Args:
        image_path (str): 图像文件的路径。
        text_labels (list): 一个包含两个描述性文本的列表，用于二分类。
                           例如：['一张正常的图片', '一张包含仇恨言论的图片']

    Returns:
        dict: 包含每个标签及其对应概率的字典。
        PIL.Image: 打开的图像对象，用于可视化。
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        return None, None

    # 打开图像
    image = Image.open(image_path)

    # 使用处理器准备数据
    # 这是CLIP的关键：它将图像和所有候选文本标签一起处理
    # padding=True 和 return_tensors="pt" 是标准操作
    inputs = processor(
        text=text_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # 模型推理 (在`no_grad`下运行以节省计算资源，因为我们只是预测，不是训练)
    with torch.no_grad():
        outputs = model(**inputs)

    # CLIP模型会为每个'文本-图像'对生成一个相似度分数 (logits)
    # logits_per_image的维度是 [1, num_text_labels]
    logits_per_image = outputs.logits_per_image

    # 使用softmax将分数转换成概率分布
    probs = logits_per_image.softmax(dim=1)

    # 将概率与标签对应起来
    results = {label: prob.item() for label, prob in zip(text_labels, probs[0])}

    return results, image


# --- 3. 结果可视化函数 ---
def visualize_result(image, results, image_path):
    """
    使用 Matplotlib 可视化分类结果。

    Args:
        image (PIL.Image): 要显示的图像。
        results (dict): 分类结果字典。
        image_path (str): 原始图像路径，用于标题。
    """
    # 找到概率最高的标签
    best_label = max(results, key=results.get)
    best_prob = results[best_label]

    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴

    # 构建标题，包含文件名和预测结果
    title = f"文件: {os.path.basename(image_path)}\n"
    title += f"预测: {best_label} ({best_prob:.2%})"

    # 在图像上方显示标题
    plt.title(title)

    # 显示图像
    plt.show()


# --- 4. 主程序入口 ---
if __name__ == "__main__":
    # 定义我们的二分类标签。CLIP通过计算图像与这些描述的相似度来进行分类
    # 这是一种 "Zero-shot" 的思想，非常灵活
    labels = ["a normal meme", "a hateful meme"]

    # --- 测试第一个样本 (非仇恨) ---
    image_file_1 = "data/not_hateful.png"
    print(f"\n--- 正在分析: {image_file_1} ---")
    results_1, image_1 = classify_meme(image_file_1, labels)

    if results_1:
        print("分析结果:")
        for label, prob in results_1.items():
            print(f"- {label}: {prob:.2%}")
        visualize_result(image_1, results_1, image_file_1)

    # --- 测试第二个样本 (仇恨) ---
    image_file_2 = "data/hateful.png"
    print(f"\n--- 正在分析: {image_file_2} ---")
    results_2, image_2 = classify_meme(image_file_2, labels)

    if results_2:
        print("分析结果:")
        for label, prob in results_2.items():
            print(f"- {label}: {prob:.2%}")
        visualize_result(image_2, results_2, image_file_2)