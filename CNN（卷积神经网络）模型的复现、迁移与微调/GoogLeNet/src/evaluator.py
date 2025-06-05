# src/evaluator.py
# 管理测试集评估和测试时增强（TTA）

import torch
import torchvision.transforms.functional as F

class Evaluator:
    def __init__(self, config, model, dataloader, dataset_size, device):
        """初始化评估器，设置模型和数据加载器"""
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.dataset_size = dataset_size
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate_with_tta(self):
        """使用 TTA 评估测试集性能"""
        self.model.eval()
        test_loss = 0.0
        test_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # TTA: 原始图像预测
                outputs_original = self.model(inputs)
                loss_val = self.criterion(outputs_original, labels)
                probs_original = torch.softmax(outputs_original, dim=1)

                # TTA: 水平翻转图像预测
                flipped_inputs = F.hflip(inputs)
                outputs_flipped = self.model(flipped_inputs)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # 平均概率
                avg_probs = (probs_original + probs_flipped) / 2
                _, preds = torch.max(avg_probs, 1)

                test_loss += loss_val.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)

        final_test_loss = test_loss / self.dataset_size
        final_test_acc = test_corrects.double() / self.dataset_size
        return final_test_loss, final_test_acc
