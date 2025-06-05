"""
TTA测试模块，负责使用测试时增强（Test-Time Augmentation）评估模型。
对测试集图像应用多种变换，取平均预测以提升性能。
"""

import torch


class TTATester:
    """负责使用TTA进行测试"""
    def __init__(self, model, test_loader, tta_transforms, device):
        self.model = model
        self.test_loader = test_loader
        self.tta_transforms = tta_transforms
        self.device = device

    def test(self):
        """使用TTA测试模型"""
        print("\n开始使用 TTA 在测试集上评估模型...")
        self.model.eval()
        total_corrects = 0
        total_images = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                image = inputs[0]
                label = labels[0].to(self.device)
                augmented_images = torch.stack([tf(image) for tf in self.tta_transforms]).to(self.device)
                outputs = self.model(augmented_images)
                mean_outputs = torch.mean(outputs, dim=0)
                _, pred = torch.max(mean_outputs, 0)
                if pred == label:
                    total_corrects += 1
                total_images += 1
        tta_acc = total_corrects / total_images
        print(f'使用 TTA 的最终测试集准确率: {tta_acc:.4f} 🎯🚀')