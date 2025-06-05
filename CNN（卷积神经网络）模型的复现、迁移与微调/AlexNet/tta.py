"""
测试时增强（TTA）模块：在测试集上使用 10-Crop TTA 策略评估模型性能。
- 对每张测试图像生成 10 个裁剪版本，聚合预测结果。
- 计算并返回测试集准确率。
"""
# tta.py
import torch
import torch.nn.functional as F
from torchvision import transforms


class TTATester:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tta_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t) for t in tensors]))
        ])

    def test(self):
        print("\n开始使用 10-Crop TTA 在测试集上评估...")
        self.model.to_device(self.device)
        self.model.eval()

        running_corrects = 0
        total_samples = len(self.test_dataset)

        for i in range(total_samples):
            image, label = self.test_dataset[i]
            tta_images = self.tta_transforms(image)
            label_tensor = torch.tensor(label, device=self.device).unsqueeze(0)
            tta_images = tta_images.to(self.device)

            with torch.no_grad():
                outputs = self.model(tta_images)
                probs = F.softmax(outputs, dim=1)
                mean_probs = torch.mean(probs, dim=0)
                _, final_pred = torch.max(mean_probs, 0)

            if final_pred == label:
                running_corrects += 1

            print(f"\r评估进度: {i + 1}/{total_samples}", end="")

        accuracy = running_corrects / total_samples
        print(f'\n\nTTA 测试集准确率: {running_corrects}/{total_samples} ({accuracy:.4f})')
        return accuracy