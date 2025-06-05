# 多模态数据分类演示（图像+文本特征融合分类基础版）

本项目演示了如何联合利用图片和文本描述，通过深度学习模型进行多模态特征提取与融合，为多模态分类任务提供基础实现框架。适合入门级学习和实验。

---

## 功能简介

- 使用 ResNet18 提取图片特征
- 使用 BERT（或 DistilBERT）提取文本特征
- 拼接图片和文本特征，实现多模态特征融合
- 输出融合后的多模态特征，为后续分类任务（如多模态商品分类等）做准备

---

## 环境依赖

- Python 3.7+
- 推荐 GPU（无 GPU 也可运行，速度略慢）
- 推荐使用虚拟环境（conda 或 venv）

### 安装依赖

```bash
pip install torch torchvision transformers pillow
```

---

## 使用方法

1. **准备测试数据**
   - 下载一张本地图片（如 `cat.jpg`），放在脚本目录下。
   - 可自定义文本描述内容。

2. **保存示例代码**（假设文件名为 `multimodal_demo.py`）：

```python
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载图片特征模型
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

# 加载文本特征模型
bert = BertModel.from_pretrained('distilbert-base-uncased').to(device).eval()
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

# 输入数据
img_path = "cat.jpg"  # 修改为你的图片路径
text = "A cat sitting on the sofa."

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

# 文本预处理
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)

# 特征提取
with torch.no_grad():
    img_feat = resnet(image)  # [1, 512]
    txt_feat = bert(**inputs).last_hidden_state.mean(dim=1)  # [1, 768]

# 融合
feature = torch.cat([img_feat, txt_feat], dim=1)
print("Multimodal feature shape:", feature.shape)
```

3. **运行脚本**

```bash
python multimodal_demo.py
```

4. **查看输出**

输出类似如下：

```
Multimodal feature shape: torch.Size([1, 1280])
```

表明已成功拼接图像和文本特征。

---

## 代码说明

- **ResNet18**：用于提取图片特征（512维向量）
- **DistilBERT**：用于提取文本特征（768维向量，取平均池化）
- **特征融合**：直接拼接图片和文本特征，得到1280维多模态特征
- **分类任务**：本代码只做特征提取和融合，如需分类，可自行添加线性层（MLP）训练和预测

---

## 可扩展方向

- 批量处理多组图片和文本，实现多模态数据集训练
- 替换为自己的数据集和类别
- 尝试其他融合方式（如加权、注意力等）
- 与 CLIP、BLIP 等预训练多模态模型对比实验

---

## 参考资料

- [PyTorch torchvision.models](https://pytorch.org/vision/stable/models.html)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/)
- [多模态学习简介](https://zhuanlan.zhihu.com/p/260703411)

---

## 联系方式

如有问题或建议，欢迎提 Issue 或联系作者。