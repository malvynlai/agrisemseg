import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

def load_dataset(data_dir, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                images.append(image)
                labels.append(label)
    
    images = torch.stack(images)
    return images, labels
    
def load_ids(csv_path):
    df = pd.read_csv(csv_path)
    ids = df['id'].tolist()
    labels = df['label'].tolist()
    return ids, labels
