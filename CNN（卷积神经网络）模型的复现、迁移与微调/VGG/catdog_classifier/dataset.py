# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    def __init__(self, dir_path, transform):
        self.dir = dir_path
        self.files = os.listdir(dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.dir, file)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = 0 if 'cat' in file.lower() else 1
        return img, label
