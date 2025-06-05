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