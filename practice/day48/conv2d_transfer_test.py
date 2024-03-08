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
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.float)


class TransferResnet18(nn.Module):
    def __init__(self):
        super(TransferResnet18, self).__init__()
        self.trsfRes = models.resnet18(pretrained=True)
        num_ftrs = self.trsfRes.fc.in_features
        self.output = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.trsfRes(x)
        x = self.output(x)
        return torch.softmax(x)
    

class make_imagelike():
    def __init__(self, data, target_size):
        self.data = data
        self.target_size = target_size

    def make_imglike(self, data, target_size):
        row = math.ceil(target_size/len(data))
        imglike_row = np.tile(data, row)[:target_size]
        imglike = np.tile(imglike_row, (target_size, 1))
        return imglike


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = { 
        'batch_size': [16, 32, 64],
        'target_size': [32, 64, 128],
        'lr': [0.001, 0.005, 0.01],
        'tuning_rate': [0.5, 0.7, 0.9]
    }


    for batch_size in parameters['batch_size']:
        for hidden_size in parameters['target_size']:
            for lr in parameters['lr']:
                for tuning_rage in parameters['tuning_rate']:
                    transforms = transforms.Compose([
                        make_imagelike()
                    ])
                    train_dataset = CustomDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    test_dataset = CustomDataset(X_test, y_test)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    model = TransferResnet18()
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()

                    trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)
                    trial.with_generators(train_loader)
                    history = trial.run(epochs=10)

                    result = trial.evaluate(data_key=torchbearer.TEST_DATA)
                    test_accuracy = result['CrossEntropyLoss']

                    print(f'Batch Size: {batch_size}, Hidden Size: {hidden_size}, Dropout Prob: {dropout_prob}, Learning Rate: {lr}')
                    print(history[-1])

                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_history = history[-1]
                        best_parameters = {'batch_size': batch_size, 'hidden_size': hidden_size, 'dropout_prob': dropout_prob, 'lr': lr}    
    print("Best Parameters:", best_parameters)
    print("Best Test Accuracy:", best_accuracy)
    print("Best Performance history", best_history)