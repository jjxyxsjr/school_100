import os
import glob
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

try:
    from model import build_vgg16_model
except ImportError:
    print("错误：无法导入 'build_vgg16_model'。请确保 model.py 文件在 Python 路径中。")
    exit()


# ==============================================================================
# ImagePredictor 类
# ==============================================================================
class ImagePredictor:
    def __init__(self, model_checkpoint_path, image_transform, class_names=None, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"图像预测器正在使用设备: {self.device}")

        self.transform = image_transform
        self.model = build_vgg16_model()

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"错误：模型检查点文件未找到: {model_checkpoint_path}")
        try:
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"模型权重已成功从 '{model_checkpoint_path}' 加载。")
        except Exception as e:
            raise RuntimeError(f"加载模型检查点时发生错误: {e}")

        self.class_names = class_names if class_names is not None else ['类别_0', '类别_1']

    def predict_image(self, image_path):
        """对单张图片进行预测。"""
        if not os.path.exists(image_path):
            print(f"错误：图片路径未找到: {image_path}")
            return None, None, None
        try:
            image_pil = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            idx = predicted_idx.item()
            name = self.class_names[idx] if self.class_names and 0 <= idx < len(self.class_names) else f"未知_{idx}"
            return name, idx, confidence.item()
        except Exception as e:
            print(f"处理或预测图片 '{image_path}' 时发生错误: {e}")
            return None, None, None

    def predict_directory(self, dir_path, show_images=True):
        """对目录中的所有图片进行预测。"""
        supported_extensions = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for ext in supported_extensions:
            image_paths.extend(glob.glob(os.path.join(dir_path, ext)))

        if not image_paths:
            print(f"在目录 '{dir_path}' 中未找到任何支持的图片。")
            return []

        print(f"在 '{dir_path}' 中找到 {len(image_paths)} 张图片，开始预测...")
        results = []
        for img_path in image_paths:
            name, _, conf = self.predict_image(img_path)
            if name is not None:
                print(
                    f"  -> 图片: {os.path.basename(img_path):<30} | 预测: {name:<10} (置信度: {conf * 100:.2f}%)")
                results.append({"path": img_path, "prediction": name, "confidence": conf})
                if show_images:
                    self.show_prediction(img_path, name, conf)
        return results

    def show_prediction(self, image_path, title, confidence):
        """使用全局设置显示中文标题。"""
        try:
            # 在这里用两行代码设置字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(f"预测: {title} (置信度: {confidence * 100:.2f}%)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"显示图片时出错: {e}")


# ==============================================================================
# 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # 1. 定义模型检查点文件的路径
    model_checkpoint = "./checkpoints/best_catdog_vgg16.pth"

    # 2. 定义图像预处理流程
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 定义类别名称
    class_names_list = ['cat', 'dog']

    # 4. 实例化 ImagePredictor
    try:
        predictor = ImagePredictor(
            model_checkpoint_path=model_checkpoint,
            image_transform=inference_transform,
            class_names=class_names_list
        )
    except Exception as e:
        print(f"初始化预测器失败: {e}")
        exit()

    # 5. 指定包含新图片的目录
    images_to_predict_dir = "./data/infer_set"

    # 6. 对目录中的图片进行预测
    if os.path.exists(images_to_predict_dir):
        prediction_results = predictor.predict_directory(images_to_predict_dir, show_images=True)
    else:
        print(f"错误：待预测图片的目录 '{images_to_predict_dir}' 未找到。")