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

device = ('cuda' if torch.cuda.is_available() else 'cpu')
pathFolder = "../earthquake/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "XTrain.pkl"
yTrainName = "yTrain.pkl"

with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

X = np.array((X*2) -1)

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


class TransferResnet18(nn.Module):
    def __init__(self, tuning_rate):
        super(TransferResnet18, self).__init__()
        self.trsfRes = models.resnet18(pretrained=False)
        num_ftrs = self.trsfRes.fc.in_features
        self.trsfRes.fc = nn.Identity()
        self.output = nn.Linear(num_ftrs, 2)

        num_params = len(list(self.trsfRes.parameters()))
        layers_to_freeze = int(num_params * tuning_rate)

        for param in list(self.trsfRes.parameters())[:layers_to_freeze]:
            param.requires_grad = False

    def forward(self, x):
        x = self.trsfRes(x)
        x = self.output(x)
        x = torch.softmax(x)
        return x


class TransferAlexnet(nn.Module):
    def __init__(self, tuning_rate):
        super(TransferAlexnet, self).__init__()
        self.trsfAlex = models.alexnet(pretrained=True)
        num_ftrs = self.trsfAlex.classifier[6].in_features
        self.trsfAlex.classifier[6] = nn.Identity()
        self.output = nn.Linear(num_ftrs, 2)

        num_params = len(list(self.trsfAlex.parameters()))
        layers_to_freeze = int(num_params * tuning_rate)

        for param in list(self.trsfAlex.parameters())[:layers_to_freeze]:
            param.requires_grad = False

    def forward(self, x):
        x = self.trsfAlex(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x    
    

def make_imglike(data, target_size):
    row = math.ceil(target_size/len(data))
    imglike_row = np.tile(data, row)[:target_size]
    imglike = np.tile(imglike_row, (3, target_size, 1))
    return imglike


if __name__ == '__main__':
    parameters = { 
        'batch_size': [16, 32, 64],
        'lr': [0.0005, 0.0007, 0.001, 0.003],
        'tuning_rate': [1]
    }

    best_accuracy = 0.0
    best_parameters = {}

    for batch_size in parameters['batch_size']:
        for lr in parameters['lr']:
            for tuning_rate in parameters['tuning_rate']:
                transform = transforms.Compose([
                    lambda x: make_imglike(x, target_size=32)
                ])

                train_dataset = CustomDataset(X_train, y_train, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                val_dataset = CustomDataset(X_val, y_val, transform=transform)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                test_dataset = CustomDataset(X_test, y_test, transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                model = TransferResnet18(tuning_rate=tuning_rate).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.BCELoss()

                trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)
                trial.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader)
                history = trial.run(epochs=20)

                result = trial.evaluate(data_key=torchbearer.TEST_DATA)
                test_accuracy = result['test_binary_acc']

                print(f'Batch Size: {batch_size}, Learning Rate: {lr}, Tuning Rate: {tuning_rate}')
                print(history[-1])

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_history = history[-1]
                    best_parameters = {'batch_size': batch_size, 'lr': lr, 'tuning_rate': tuning_rate}
                    torch.save(model, './conv2d_Resnet18_no_weight_best.pt')

    print("Best Parameters:", best_parameters)
    print("Best Test Accuracy:", best_accuracy)
    print("Best Performance history", best_history)