import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torchvision.models import resnet50
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.utils.data
from torch.utils.data import DataLoader

class ResNet50Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=1, learning_rate=0.001, epochs=5, batch_size=32):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)


    def _build_model(self):
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.n_classes)
        return model.to(self.device)


    def fit(self, X, y):
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            print(f'current epoch {epoch}')
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self


    def predict(self, X):
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)
                outputs = self.model(batch_X)
                preds = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(preds)
        return np.array(predictions)
