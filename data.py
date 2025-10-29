import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class AcneDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, **kwargs):
        # Silence Keras warning
        if 'workers' in kwargs:
            pass
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label2id = {lbl:i for i,lbl in enumerate(sorted(self.df['label'].unique()))}
        self.id2label = {v:k for k,v in self.label2id.items()}

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['file_path'])
        image = Image.open(img_path).convert("RGB")
        label = self.label2id[row['label']]
        if self.transform: image = self.transform(image)
        return image, label

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
