import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from torchbearer import Trial
from sklearn.model_selection import train_test_split

pathFolder = "./train/spaceship/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "XTrain.pkl"
yTrainName = "yTrain.pkl"

with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

print(X[0], y[0])
print(type(X),type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.float)
    

if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = { 
        'batch_size': [8, 16, 32],
        'hidden_size': [32, 64, 128],
        'dropout_prob': [0.2, 0.5, 0.8],
        'lr': [0.001, 0.01, 0.1]
    }


    for batch_size in parameters['batch_size']:
        for hidden_size in parameters['hidden_size']:
            for dropout_prob in parameters['dropout_prob']:
                for lr in parameters['lr']:

                    train_dataset = CustomDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    test_dataset_fc = CustomDataset(X_test, y_test)
                    test_loader_fc = DataLoader(test_dataset_fc, batch_size=batch_size, shuffle=False)

                    model = BinaryClassificationModel(input_size = 4, hidden_size=hidden_size, dropout_prob=dropout_prob)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.BCELoss()

                    trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)
                    trial.with_generators(train_loader)
                    history = trial.run(epochs=10)

                    print(f'Batch Size: {batch_size}, Hidden Size: {hidden_size}, Dropout Prob: {dropout_prob}, Learning Rate: {lr}')
                    print(history[-1])