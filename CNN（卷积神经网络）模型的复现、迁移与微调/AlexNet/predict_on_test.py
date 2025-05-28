# predict_on_test.py

import torch
import torch.nn as nn
import os
import json
from torchvision import models as tv_models, transforms
from PIL import Image  # 用于加载单个图像
import pandas as pd  # 用于创建预测结果的CSV文件

# --- 配置 ---
MODEL_PATH = "alexnet_flowers_best_feature_extract.pth"  # 你训练好的最佳模型
BASE_DATA_DIR = r"D:\PyCharm\Py_Projects\school\school_100\CNN（卷积神经网络）模型的复现、迁移与微调\AlexNet\data\dataset"
TEST_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "test")  # 包含测试图像的扁平化目录
CAT_TO_NAME_PATH = os.path.join(os.path.dirname(BASE_DATA_DIR), "cat_to_name.json")  # 类别到名称映射文件的路径

NUM_CLASSES = 102  # 必须与训练模型时的类别数一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV_PATH = "test_predictions.csv"

# ImageNet的均值和标准差，用于归一化（必须与训练时一致）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224  # AlexNet 输入尺寸


def load_class_mappings(train_data_dir_for_classes, cat_to_name_json_path):
    """
    加载映射关系：
    1. idx_to_class_folder_name: 从整数索引 (0-101) 到原始类别文件夹名称 (例如, '1', '10')。
       这是通过从训练数据结构创建一个临时的 ImageFolder 数据集来派生的。
    2. class_folder_name_to_flower_name: 从类别文件夹名称到实际花卉名称 (例如, '1' -> '月季')。
       这是从 cat_to_name.json 加载的。
    """
    # 1. 获取 idx_to_class_folder_name (从训练集结构)
    # 我们需要训练时使用的 ImageFolder 数据集实例的 `classes` 属性
    # 来了解模型输出索引 (0-101) 到原始类别文件夹名称 ('1', '10' 等) 的映射
    idx_to_class_folder_name = {}
    try:
        # 创建一个临时的变换，我们只需要 'classes' 属性
        dummy_transform = transforms.Compose([transforms.Resize(INPUT_SIZE), transforms.ToTensor()])
        # 指向训练数据目录以获取训练时建立的类别名称
        # 假设你的 `train` 文件夹下有 '1', '10' 等类别子文件夹
        train_class_structure_dir = os.path.join(os.path.dirname(TEST_IMAGES_DIR), "train")  # 假设 test 和 train 是同级目录
        if not os.path.exists(train_class_structure_dir):
            print(f"警告：用于类别映射的训练目录 '{train_class_structure_dir}' 未找到。预测将使用数字类别ID。")
        else:
            temp_train_dataset = transforms.datasets.ImageFolder(train_class_structure_dir, transform=dummy_transform)
            # temp_train_dataset.classes 是一个类似 ['1', '10', '100', ..., '99'] 的列表
            # 此列表中的索引对应于模型的输出类别索引
            idx_to_class_folder_name = {i: class_name for i, class_name in enumerate(temp_train_dataset.classes)}
            if len(idx_to_class_folder_name) != NUM_CLASSES:
                print(f"警告：从训练目录获取的类别数量 ({len(idx_to_class_folder_name)}) "
                      f"与 NUM_CLASSES ({NUM_CLASSES}) 不匹配。如果映射失败，将使用直接的整数索引。")
                idx_to_class_folder_name = {}  # 如果不一致则重置

    except Exception as e:
        print(f"无法从训练目录结构加载类别文件夹名称: {e}")
        print("预测将使用数字类别ID代替文件夹名称。")

    # 2. 加载 class_folder_name_to_flower_name (从 cat_to_name.json)
    class_folder_name_to_flower_name = {}
    try:
        with open(cat_to_name_json_path, 'r', encoding='utf-8') as f:  # 指定utf-8编码
            class_folder_name_to_flower_name = json.load(f)
    except FileNotFoundError:
        print(f"警告：在 {cat_to_name_json_path} 未找到 cat_to_name.json。无法映射到花卉名称。")
    except json.JSONDecodeError:
        print(f"警告：解码 {cat_to_name_json_path} 出错。无法映射到花卉名称。")

    return idx_to_class_folder_name, class_folder_name_to_flower_name


def preprocess_image(image_path):
    """加载图像并应用必要的变换。"""
    image = Image.open(image_path).convert('RGB')  # 确保是3通道图像

    # 变换 (必须与训练时的验证/测试变换一致)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return transform(image).unsqueeze(0)  # 添加批次维度


if __name__ == '__main__':
    print(f"预测将在设备: {DEVICE} 上运行")
    print(f"从以下路径加载模型: {MODEL_PATH}")
    print(f"处理以下路径的图像: {TEST_IMAGES_DIR}")

    # --- 1. 加载映射关系 ---
    idx_to_class_folder_name, class_folder_name_to_flower_name = load_class_mappings(
        os.path.join(BASE_DATA_DIR, "train"),  # 指向你的训练数据文件夹以获取类别结构
        CAT_TO_NAME_PATH
    )
    if not idx_to_class_folder_name:  # 如果从训练结构映射失败，则回退
        print("由于文件夹名称映射失败，将使用直接的整数索引 (0-101) 作为类别标识符。")
        idx_to_class_folder_name = {i: str(i) for i in range(NUM_CLASSES)}  # 例如, 0 -> "0"

    # --- 2. 加载模型 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件 '{MODEL_PATH}' 未找到。")
        exit()

    model_to_predict = tv_models.alexnet(weights=None)
    num_ftrs = model_to_predict.classifier[6].in_features
    model_to_predict.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

    try:
        model_to_predict.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"加载模型权重出错: {e}")
        exit()

    model_to_predict = model_to_predict.to(DEVICE)
    model_to_predict.eval()  # 设置为评估模式
    print("模型加载成功。")

    # --- 3. 获取测试图像列表 ---
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"错误: 测试图像目录 '{TEST_IMAGES_DIR}' 未找到。")
        exit()

    try:
        image_filenames = [f for f in os.listdir(TEST_IMAGES_DIR) if os.path.isfile(os.path.join(TEST_IMAGES_DIR, f))
                           and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not image_filenames:
            print(f"在 '{TEST_IMAGES_DIR}' 中未找到图像文件。")
            exit()
        print(f"在测试目录中找到 {len(image_filenames)} 张图像。")
    except Exception as e:
        print(f"列出测试目录中的文件时出错: {e}")
        exit()

    # --- 4.进行预测 ---
    predictions_data = []
    print("\n正在进行预测...")
    with torch.no_grad():
        for filename in image_filenames:
            image_path = os.path.join(TEST_IMAGES_DIR, filename)
            try:
                image_tensor = preprocess_image(image_path)
                image_tensor = image_tensor.to(DEVICE)

                outputs = model_to_predict(image_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_idx_item = predicted_idx.item()  # 获取整数索引

                # 将预测索引映射到原始类别文件夹名称 (例如, '1', '10')
                # 这依赖于 idx_to_class_folder_name 是否被正确填充
                predicted_class_folder_name = idx_to_class_folder_name.get(predicted_idx_item, str(predicted_idx_item))

                # 将类别文件夹名称映射到实际花卉名称
                predicted_flower_name = class_folder_name_to_flower_name.get(predicted_class_folder_name,
                                                                             "未知")  # "Unknown"

                predictions_data.append({
                    'image_filename': filename,
                    'predicted_class_id_numeric': predicted_idx_item,  # 0-101
                    'predicted_class_folder_name': predicted_class_folder_name,  # '1', '10', 等.
                    'predicted_flower_name': predicted_flower_name
                })
                if len(predictions_data) % 50 == 0:  # 打印进度
                    print(f"  已处理 {len(predictions_data)}/{len(image_filenames)} 张图像...")

            except Exception as e:
                print(f"处理图像 {filename} 时出错: {e}")
                predictions_data.append({
                    'image_filename': filename,
                    'predicted_class_id_numeric': -1,
                    'predicted_class_folder_name': "错误",
                    'predicted_flower_name': "处理错误"
                })

    print(f"对 {len(predictions_data)} 张图像的预测完成。")

    # --- 5. 保存预测结果到CSV ---
    if predictions_data:
        pred_df = pd.DataFrame(predictions_data)
        try:
            pred_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')  # utf-8-sig 确保中文在Excel中正确显示
            print(f"\n预测结果已保存到: {OUTPUT_CSV_PATH}")
        except Exception as e:
            print(f"保存预测结果到CSV时出错: {e}")
    else:
        print("没有可保存的预测结果。")