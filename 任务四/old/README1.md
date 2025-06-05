# 文本分类与生成演示（Transformers基础版）

本项目提供了一个简单易用的文本情感分析与文本生成的演示脚本，基于 [Hugging Face Transformers](https://huggingface.co/docs/transformers/) 库。  
适合初学者体验自然语言处理（NLP）基础任务。

---

## 主要功能

- **文本情感分析**：输入若干英文句子，自动输出积极/消极分类及置信分数
- **文本生成**：输入一句开头，自动用GPT-2续写句子

---

## 运行环境要求

- Python 3.7+
- 推荐使用虚拟环境（如conda或venv）

### 依赖安装

```bash
pip install transformers torch
```

---

## 使用方法

1. **保存你的主代码**（如下，假设为 `text_demo.py`）：

```python
from transformers import pipeline

def task_text_classification():
    # ...（此处省略，用你自己的主代码即可）...

def task_text_generation():
    # ...（此处省略，用你自己的主代码即可）...

if __name__ == "__main__":
    task_text_classification()
    task_text_generation()
```

2. **运行代码**

```bash
python text_demo.py
```

3. **查看输出**

程序会自动下载所需模型，并输出类似如下结果：

```
--- 任务 1.1: 文本分类 (基础版) ---
正在加载情感分析模型...
模型加载完成！
正在进行预测...
文本: 'This is such a beautiful day!'
  -> 标签: POSITIVE, 分数: 0.9998
文本: 'I am not happy with the service.'
  -> 标签: NEGATIVE, 分数: 0.9992

--- 任务 1.2: 文本生成 (基础版) ---
正在加载 GPT-2 模型...
模型加载完成！
正在根据 'In a world where AI is taking over,' 生成文本...
--------------------
In a world where AI is taking over, ...（后续生成内容）
```

---

## 代码说明

- **task_text_classification()**  
  使用微调过的DistilBERT模型做英文情感分类（积极/消极），支持批量文本。
- **task_text_generation()**  
  使用经典GPT-2模型做英文文本生成，可自定义开头和生成长度。
- **framework='pt'**  
  显式指定使用PyTorch框架，兼容性好。

---

## 常见问题

- **模型首次运行会自动下载，速度取决于网络情况**
- **如遇GPU/显存不足，可自动转为CPU运行**
- **支持自定义输入文本和生成开头**

---

## 参考资料

- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/)
- [DistilBERT 情感分类模型](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [GPT-2 文本生成模型](https://huggingface.co/gpt2)

---

## 联系方式

如有问题或建议，欢迎提 Issue 或联系作者。