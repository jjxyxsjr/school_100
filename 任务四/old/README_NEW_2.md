# 多模态数据分类演示（图像+文本分类基础版）

本项目提供一个简单的多模态数据分类（图像+文本）代码示例，适合入门级学习和实验。  
你可以体验到如何联合利用图片与文字描述，实现多模态特征融合，为后续分类任务打下基础。

---

## 主要功能

- 使用 ResNet18 提取图像特征
- 使用 BERT（如 DistilBERT）提取文本特征
- 拼接多模态特征，为分类器做准备
- 适合用于多模态分类、融合实验等

---

## 运行环境要求

- Python 3.7+
- 推荐 GPU（无 GPU 也可，速度略慢）
- 推荐使用虚拟环境（如 conda 或 venv）

### 依赖安装

```bash
pip install torch torchvision transformers pillow
```

---

## 使用方法

1. **保存代码**（假设文件名为 `multimodal_demo.py`）：

```python
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()
bert = BertModel.from_pretrained('distilbert-base-uncased').to(device).eval()
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

# 输入数据
img_path = "cat.jpg"  # 换成你自己的图片路径
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

2. **准备测试数据**  
   - 下载一张本地图片（如 `cat.jpg`），放在与脚本相同目录。
   - 可将 `text` 修改为你的自定义描述。

3. **运行脚本**

```bash
python multimodal_demo.py
```

4. **查看输出**

输出类似如下：

```
Multimodal feature shape: torch.Size([1, 1280])
```
说明图片特征和文本特征已成功拼接，后续即可接入分类器进行多模态分类。

---

## 代码说明

- **ResNet18**：预训练模型，提取图片特征（512维）。
- **DistilBERT**：提取文本特征（768维，取平均池化）。
- **特征融合**：直接拼接成1280维特征向量。
- **分类任务**：此脚本只做特征提取和融合。如需分类，可自行添加线性层（MLP）进行训练和预测。

---

## 拓展建议

- 可批量处理多组“图片+文本”，用于训练多模态分类网络。
- 可替换为自己的数据集、类别和更复杂的融合方式。
- 进一步可与CLIP等预训练多模态模型对比实验。

---

## 参考资料

- [PyTorch torchvision.models](https://pytorch.org/vision/stable/models.html)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/)
- [多模态学习简介（知乎专栏）](https://zhuanlan.zhihu.com/p/260703411)

---

## 联系方式

如有问题或建议，欢迎提 Issue 或联系作者。