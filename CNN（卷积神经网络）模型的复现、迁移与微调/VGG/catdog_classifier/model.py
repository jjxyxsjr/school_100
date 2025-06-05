# model.py (使用您期望的新结构)

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

def build_vgg16_model():
    vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
    for param in vgg16.parameters():
        param.requires_grad = False

    # 使用您提出的新分类头结构
    num_ftrs = vgg16.classifier[0].in_features
    vgg16.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 2048),  # 您期望的结构
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 2)      # 您期望的结构
    )
    return vgg16

# ... (unfreeze_all_layers 函数保持不变) ...
def unfreeze_all_layers(model):
    print("\n...正在解冻模型的所有层用于微调...")
    for name, param in model.named_parameters():
        param.requires_grad = True