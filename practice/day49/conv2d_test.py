import numpy as np
import os
import math
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torchbearer import Trial
import torchbearer
from sklearn.model_selection import train_test_split

pathFolder = "../day47/train/spaceship/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "XTrain.pkl"
yTrainName = "yTrain.pkl"

with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imglike_data = torch.tensor(self.data[idx], dtype=torch.float32)

        if self.transform:
            imglike_data = self.transform(imglike_data)
            
        return imglike_data, torch.tensor(self.labels[idx], dtype=torch.float32)


class conv2d_model(nn.Module):
    def __init__(self, tuning_rate):
        super(conv2d_model, self).__init__()
        self.trsfRes = models.resnet18(pretrained=True)
        num_ftrs = self.trsfRes.fc.in_features
        self.trsfRes.fc = nn.Identity()
        self.output = nn.Linear(num_ftrs, 1)

        num_params = len(list(self.trsfRes.parameters()))
        layers_to_freeze = int(num_params * tuning_rate)

        for param in list(self.trsfRes.parameters())[:layers_to_freeze]:
            param.requires_grad = False

    def forward(self, x):
        x = self.trsfRes(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
    

def make_imglike(data, target_size):
    row = math.ceil(target_size/len(data))
    imglike_row = np.tile(data, row)[:target_size]
    imglike = np.tile(imglike_row, (3, target_size, 1))
    return imglike


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')