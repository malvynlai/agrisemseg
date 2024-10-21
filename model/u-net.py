import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


output_dir = 'trial_outputs'
os.makedirs(output_dir, exist_ok=True)

import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_model='resnet18'):
        super(UNet, self).__init__()
        if resnet_model == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif resnet_model == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif resnet_model == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError('Invalid ResNet model specified.')

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, label_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.images = []
        self.labels = []

        for filename in os.listdir(os.path.join(root_dir, "images", "rgb")):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(root_dir, "images", "rgb", filename)
                label_path = os.path.join(root_dir, "labels", "double_plant", filename.split('.')[0] + '.png')
                print(image_path, label_path)
                self.images.append(image_path)
                self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return image, label

root_dir = '/home/ubuntu/supervised/Agriculture-Vision-2021/val'
batch_size = 1
num_workers = 0
learning_rate = 0.001
num_epochs = 1

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # change this
])
label_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SatelliteDataset(root_dir, image_transform, label_transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

in_channels = 3  
out_channels = 1  
model = UNet(in_channels, out_channels)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print("------Running Epoch------")
    counter = 0
    for images, labels in train_loader:
        print("inside loop", counter)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output_filename = f'ing_{counter+1}_epoch_{epoch+1}.png'
        output_path = os.path.join(output_dir, output_filename)
        save_image(outputs, output_path)
        counter += 1

    

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
torch.save(model.state_dict(), 'trained_model.pth')