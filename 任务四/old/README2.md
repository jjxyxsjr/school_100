# 多模态图文匹配演示 (CLIP 基础版)

本项目提供了一个基于 [Hugging Face Transformers](https://huggingface.co/docs/transformers/) 和 OpenAI CLIP 模型的“图文匹配”演示代码。  
你可以用它体验 AI 如何理解图片和文本描述之间的语义相关性。

---

## 主要功能

- **下载并加载 CLIP 预训练模型（`openai/clip-vit-base-patch32`）**
- **自动下载一张网络图片，并与多条文本描述进行语义匹配**
- **输出每条描述与图片的匹配概率，帮助你理解 AI 的多模态理解能力**

---

## 运行环境要求

- Python 3.7+
- 推荐显卡（无显卡用 CPU 也可运行，速度略慢）
- 推荐使用虚拟环境（如conda或venv）

### 依赖安装

```bash
pip install transformers torch pillow requests
```

---

## 使用方法

1. **保存你的主代码**（如下，假设为 `clip_demo.py`）：

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

def task_multimodal_classification():
    # ...（此处省略，用你自己的主代码即可）...

if __name__ == "__main__":
    task_multimodal_classification()
```

2. **运行代码**

```bash
python clip_demo.py
```

3. **查看输出**

程序会自动下载模型和图片，输出类似如下结果：

```
--- 匹配结果 ---
  -> 描述: a photo of a cat sleeping on a bed            | 匹配概率: 0.0721
  -> 描述: two cats looking at a television screen       | 匹配概率: 0.9251
  -> 描述: a photo of a dog playing fetch                | 匹配概率: 0.0019
  -> 描述: an astronaut on the moon                      | 匹配概率: 0.0009
```

---

## 代码说明

- **模型加载**：首次运行会自动下载 CLIP 预训练模型（约1.7GB），请确保网络畅通。
- **图片与文本描述**：你可以修改图片 URL 和文本描述列表，体验不同内容的匹配效果。
- **输出概率**：概率越高，说明该描述与图片越匹配。

---

## 常见问题

- **模型下载慢/失败？**  
  请检查网络，或提前科学上网。
- **Windows 下报 symlink 警告？**  
  不影响功能，可忽略。
- **显存不足？**  
  该模型对硬件要求较低，普通笔记本/CPU也能运行。

---

## 参考资料

- [CLIP 官方模型文档](https://huggingface.co/docs/transformers/model_doc/clip)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

---

## 联系作者

如有问题或建议，欢迎提 Issue 或联系作者。
