import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 1. 设置随机种子
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 2. 构造模拟数据集
# 随机生成图片（3x224x224），随机中文文本，二分类标签
CHINESE_TEXTS = [
    "这只猫很可爱。",
    "天气真好。",
    "我喜欢看书。",
    "这条狗跑得很快。",
    "电脑出现了故障。",
    "他今天心情不错。",
    "下雨了，出门记得带伞。",
    "我喜欢吃苹果。",
    "小明在操场上踢足球。",
    "老师正在上课。"
]
N_SAMPLES = 100  # 数据量

data = []
for i in range(N_SAMPLES):
    # 随机图片
    img = torch.rand(3, 224, 224)
    # 随机文本
    text = random.choice(CHINESE_TEXTS)
    # 随机标签
    label = random.randint(0, 1)
    data.append({"image": img, "text": text, "label": label})

# 3. Dataset类
class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"])
        text = item["text"]
        label = item["label"]
        return img, text, label

dataset = MultiModalDataset(data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# 4. 加载中文BERT 文本塔
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').to(device)
bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
bert_model.eval()

# 5. 加载ResNet18 视觉塔
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # 只要特征，不要最后分类层
resnet = resnet.to(device)
resnet.eval()

# 6. 分类器
class MultimodalClassifier(nn.Module):
    def __init__(self, img_dim=512, text_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, img_feat, text_feat):
        fused = torch.cat([img_feat, text_feat], dim=1)
        out = self.mlp(fused)
        return out

model = MultimodalClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 7. 特征提取函数
@torch.no_grad()
def extract_img_feat(img_batch):
    return resnet(img_batch)

@torch.no_grad()
def extract_text_feat(text_batch):
    # text_batch: list of str
    tokens = bert_tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=32)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = bert_model(**tokens)
    # 取[CLS]的特征
    return out.last_hidden_state[:, 0, :]

# 8. 训练与评估
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total, correct, train_loss = 0, 0, 0
    for img, text, label in train_loader:
        img, label = img.to(device), label.to(device)
        img_feat = extract_img_feat(img)
        text_feat = extract_text_feat(list(text))
        logits = model(img_feat, text_feat)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += img.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/total:.4f} | Train Acc: {acc:.4f}")

    # 测试
    model.eval()
    total, correct, test_loss = 0, 0, 0
    with torch.no_grad():
        for img, text, label in test_loader:
            img, label = img.to(device), label.to(device)
            img_feat = extract_img_feat(img)
            text_feat = extract_text_feat(list(text))
            logits = model(img_feat, text_feat)
            loss = criterion(logits, label)
            test_loss += loss.item() * img.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += img.size(0)
    acc = correct / total
    print(f"         | Test  Loss: {test_loss/total:.4f} | Test  Acc: {acc:.4f}")