import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
"""

"""
class AgriDataset(Dataset):
    def __init__(self, label_dir, image_dir, transform=None):
        self.labels = pd.read_csv(label_dir, header=None)
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(self.image_dir)
        print(len(self.image_files))
        print(len(self.labels))
        assert len(self.image_files) == len(self.labels), "Number of images does not match number of labels"
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(img_name).convert("RGB")
        labels = self.labels.iloc[index].values
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels