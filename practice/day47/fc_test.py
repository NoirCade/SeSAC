import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from torchbearer import Trial
import torchbearer
from sklearn.model_selection import train_test_split


'''추후 적용해볼 것들
    1. BCE - sigmoid 대신 CEL - softmax
'''


pathFolder = "./train/spaceship/"
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
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.float)


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(BinaryClassificationModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)  # 선형 레이어
        self.hidden = nn.Linear(hidden_size, hidden_size)   # 중간 레이어
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # 배치 정규화 레이어
        self.dropout = nn.Dropout(dropout_prob)  # 드롭아웃 레이어
        self.output = nn.Linear(hidden_size, 1)  # 출력 레이어

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.output(x)

        return torch.sigmoid(x)


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = { 
        'batch_size': [4, 8, 16],
        'hidden_size': [16, 32, 64],
        'dropout_prob': [0.03, 0.05, 0.07],
        'lr': [0.01, 0.02, 0.03]
    }

    best_accuracy = 0.0
    best_parameters = {}

    for batch_size in parameters['batch_size']:
        for hidden_size in parameters['hidden_size']:
            for dropout_prob in parameters['dropout_prob']:
                for lr in parameters['lr']:

                    train_dataset = CustomDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    val_dataset = CustomDataset(X_val, y_val)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                    test_dataset = CustomDataset(X_test, y_test)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    model = BinaryClassificationModel(input_size = 4, hidden_size=hidden_size, dropout_prob=dropout_prob)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.BCELoss()

                    trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)
                    trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
                    history = trial.run(epochs=30)

                    result = trial.evaluate(data_key=torchbearer.TEST_DATA)
                    test_accuracy = result['test_binary_acc']

                    print(f'Batch Size: {batch_size}, Hidden Size: {hidden_size}, Dropout Prob: {dropout_prob}, Learning Rate: {lr}')
                    print(history[-1])

                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_history = history[-1]
                        best_parameters = {'batch_size': batch_size, 'hidden_size': hidden_size, 'dropout_prob': dropout_prob, 'lr': lr}
                        torch.save(model, './fc_best_model.pt')


    print("Best Parameters:", best_parameters)
    print("Best Test Accuracy:", best_accuracy)
    print("Best Performance history", best_history)