from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests


def task_multimodal_classification():
    """使用基础版 CLIP 模型进行图文匹配"""
    print("--- 任务 2: 图文匹配 (基础版) ---")

    # 定义基础版模型ID，这个模型不需要登录，对硬件要求较低
    model_id = "openai/clip-vit-base-patch32"

    # 分别加载处理器(Processor)和模型(Model)
    # 处理器负责将图片和文本转换成模型能理解的格式
    print(f"首次运行时将下载模型: {model_id} (约1.7GB)...")
    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
    except Exception as e:
        print(f"模型下载或加载失败，请检查网络连接。错误: {e}")
        return

    print("模型加载完成！")

    # 准备一张网络图片
    # 这是一张著名的"两次猫在沙发上看电视"的图片
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 准备几句候选的文字描述
    text_options = [
        "a photo of a cat sleeping on a bed",
        "two cats looking at a television screen",
        "a photo of a dog playing fetch",
        "an astronaut on the moon"  # 这是错误的描述，用来对比
    ]

    print(f"\n正在下载并分析图片: {image_url}")
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except requests.exceptions.RequestException as e:
        print(f"无法下载图片，请检查网络连接或URL: {e}")
        return

    print("正在计算图片与文本的相似度...")
    # 使用处理器来准备输入数据，并用模型进行计算
    # return_tensors="pt" 表示返回 PyTorch Tensors
    inputs = processor(text=text_options, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # logits_per_image 包含了图片与每句文本的原始匹配分数
    logits_per_image = outputs.logits_per_image

    # 我们使用 softmax 函数将原始分数转换成和为1的概率，更便于理解
    probs = logits_per_image.softmax(dim=1)

    # --- 打印结果 ---
    print("\n--- 匹配结果 ---")
    # tolist()[0] 将PyTorch Tensor转换为Python列表以便处理
    for text, prob in zip(text_options, probs.tolist()[0]):
        # 使用 f-string 格式化输出，让结果对齐，更美观
        print(f"  -> 描述: {text:<45} | 匹配概率: {prob:.4f}")


if __name__ == "__main__":
    task_multimodal_classification()

    # 你的CLIP代码
    # 模型：CLIP（Contrastive Language - ImagePre - training），由OpenAI提出，是目前最经典的多模态对齐模型。
    # 任务：图文匹配（给一张图片，和多句文本，输出每个文本和图片的匹配概率）。
    # 输入：一张图片 + 多个文本描述。
    # 输出：每个文本与图片的匹配分数（概率）。
    # 用途：常用于图文检索、自动标注、内容理解等。
    # 优势：CLIP模型已在大规模数据上对齐学到“图片 - 文本”语义空间，效果非常好，且用法极其简单。
    # 适合初学者直接体验AI理解多模态的"语义对齐"能力。
    # 总结建议
    # 如果你只是想体验多模态AI理解“图 + 文”语义匹配、检索的强大，CLIP是最推荐的方案，代码短、效果好。
    # 如果你要自定义多模态分类任务，比如“给图片和描述，分辨是宠物还是风景”，或者想玩特征拼接 / 多模态融合实验，则用multimodal_demo.py，并自己补个分类头。
