# predictor.py

import os
import glob
import torch
# from torchvision import transforms # transforms 由调用方传入 __init__
from PIL import Image

# predictor.py 需要能够导入您的模型定义
# 假设 model.py 和 predictor.py 在同一个项目文件夹或可访问的路径中
from model import build_vgg16_model  # 确保 model.py 中有此函数


class ImagePredictor:
    def __init__(self, model_checkpoint_path, image_transform, class_names=None, device=None):
        """
        初始化图像预测器。

        参数:
            model_checkpoint_path (str): 已训练模型权重文件的路径。
            image_transform (callable): 应用于输入图像的 torchvision 转换。
            class_names (list, optional): 类别名称列表，顺序应与模型输出对应。
                                         如果为 None，将使用通用名称或尝试后续推断。
            device (str, optional): 指定设备 ('cuda', 'cpu')。如果为 None，则自动检测。
        """
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"图像预测器正在使用设备: {self.device}")

        self.transform = image_transform

        # 1. 构建模型结构
        self.model = build_vgg16_model()  # 这会创建VGG16的网络结构

        # 2. 加载训练好的权重
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"错误：模型检查点文件未找到: {model_checkpoint_path}")
        try:
            # 加载状态字典到当前设备
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
            self.model.to(self.device)  # 将整个模型移动到设备
            self.model.eval()  # 关键：将模型设置为评估模式
            print(f"模型权重已成功从 '{model_checkpoint_path}' 加载。")
        except Exception as e:
            raise RuntimeError(f"加载模型检查点 '{model_checkpoint_path}' 时发生错误: {e}")

        # 3. 设置类别名称
        self.class_names = class_names
        if self.class_names is None:
            try:
                num_outputs = self.model.classifier[-1].out_features
                self.class_names = [f'类别_{i}' for i in range(num_outputs)]
                print(f"未提供类别名称。已生成通用类别名称: {self.class_names}")
            except Exception as e:
                print(f"警告：无法从模型结构确定输出类别数以生成通用名称: {e}")
                self.class_names = ['类别_0', '类别_1']
                print(f"已回退至默认类别名称: {self.class_names}")

    def set_class_names_from_train_dir(self, train_dir):
        """
        (可选方法) 尝试从训练数据目录结构推断类别名称并更新 self.class_names。
        这假设训练目录的子文件夹名称对应类别，并按字母顺序排序。

        参数:
            train_dir (str): 原始训练数据的根目录。
        """
        print(f"正在尝试从 '{train_dir}' 推断类别名称...")
        if train_dir and os.path.exists(train_dir):
            try:
                potential_classes = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
                num_model_outputs = self.model.classifier[-1].out_features
                if potential_classes and len(potential_classes) == num_model_outputs:
                    self.class_names = potential_classes
                    print(f"成功从 '{train_dir}' 推断类别名称: {self.class_names}")
                    return True
                else:
                    reason = "未找到子目录" if not potential_classes else \
                        f"子目录数({len(potential_classes)})与模型输出数({num_model_outputs})不匹配"
                    print(f"警告: 未能从 '{train_dir}' 可靠推断类别名称。原因: {reason}。")
            except Exception as e:
                print(f"警告: 从 '{train_dir}' 推断类别名称时发生错误: {e}")
        else:
            print(f"警告: 提供的训练目录 '{train_dir}' 无效或不存在。")

        print("类别名称未通过目录推断更新。将继续使用现有或通用类别名称。")
        return False

    def predict_image(self, image_path):
        """
        对单张图片进行预测。

        参数:
            image_path (str): 待预测图片的路径。

        返回:
            tuple: (predicted_class_name, predicted_class_idx, confidence_score)
                   如果出错则返回 (None, None, None)。
        """
        if not os.path.exists(image_path):
            print(f"错误：图片路径未找到: {image_path}")
            return None, None, None

        try:
            image_pil = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image_pil)
            img_batch_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_logits = self.model(img_batch_tensor)
                probabilities = torch.softmax(output_logits, dim=1)
                confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 1)

                predicted_class_idx = predicted_idx_tensor.item()
                confidence_score = confidence_tensor.item()

                predicted_class_name = "未知类别"
                if self.class_names and 0 <= predicted_class_idx < len(self.class_names):
                    predicted_class_name = self.class_names[predicted_class_idx]
                else:
                    print(
                        f"警告: 预测索引 {predicted_class_idx} 超出类别名称列表范围 (长度 {len(self.class_names) if self.class_names else 0})。")
                    predicted_class_name = f"类别索引_{predicted_class_idx}"

            return predicted_class_name, predicted_class_idx, confidence_score
        except Exception as e:
            print(f"处理或预测图片 '{image_path}' 时发生错误: {e}")
            return None, None, None

    def predict_directory(self, dir_path, show_images_on_console=False):
        """
        对指定目录下的所有支持的图片进行预测。

        参数:
            dir_path (str): 包含待预测图片的目录路径。
            show_images_on_console (bool): 是否尝试使用matplotlib显示每张图片及其预测结果。

        返回:
            list: 包含每个图片预测结果的字典列表。
        """
        if not os.path.exists(dir_path):
            print(f"错误：目录 '{dir_path}' 未找到。")
            return []

        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        image_paths = []
        for ext in supported_extensions:
            image_paths.extend(glob.glob(os.path.join(dir_path, ext)))

        if not image_paths:
            print(f"在目录 '{dir_path}' 中未找到任何支持的图片文件。")
            return []

        print(f"\n在目录 '{dir_path}' 中找到 {len(image_paths)} 张图片。开始预测...\n")

        all_results = []
        # 仅当需要显示图片时，才进行一次字体配置的尝试
        # 这个布尔值用于确保字体配置代码只在第一次需要显示图片时执行一次（或者每次都执行，取决于你是否想每次都打印提示）
        matplotlib_chinese_configured = False

        for img_path in image_paths:
            predicted_name, _, confidence = self.predict_image(img_path)

            if predicted_name is not None:
                print(
                    f"  图片: {os.path.basename(img_path):<30} -> 预测: {predicted_name:<10} (置信度: {confidence * 100:.2f}%)")
                all_results.append({
                    "path": img_path,
                    "prediction": predicted_name,
                    "confidence": confidence
                })
                if show_images_on_console:
                    try:
                        import matplotlib.pyplot as plt

                        # --- 开始：添加中文支持配置 ---
                        # 仅在首次尝试显示图片时配置一次，或根据需要调整此逻辑
                        if not matplotlib_chinese_configured:
                            font_list_for_chinese = [
                                'SimHei', 'Microsoft YaHei', 'DengXian',
                                'PingFang SC', 'Heiti SC',
                                'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                                'sans-serif'
                            ]
                            plt.rcParams['font.sans-serif'] = font_list_for_chinese
                            plt.rcParams['axes.unicode_minus'] = False
                            print("提示: 已尝试配置matplotlib中文字体支持。若仍有问题，请确保已安装推荐字体之一。")
                            matplotlib_chinese_configured = True  # 标记已配置
                        # --- 结束：添加中文支持配置 ---

                        image_pil_to_show = Image.open(img_path).convert('RGB')
                        plt.imshow(image_pil_to_show)
                        plt.title(f"预测: {predicted_name} ({confidence * 100:.2f}%)", fontsize=12)
                        plt.axis('off')
                        plt.show()
                    except ImportError:
                        print("提示: 未安装 Matplotlib，无法显示图片。如果需要此功能，请安装它。")
                        show_images_on_console = False
                    except Exception as e:
                        print(f"显示图片 '{img_path}' 时发生错误: {e}")
            else:
                all_results.append({
                    "path": img_path,
                    "prediction": "错误",
                    "confidence": 0.0
                })
        return all_results