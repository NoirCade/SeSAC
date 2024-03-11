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
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split


'''추후 적용해볼 것들
    1. BCE - sigmoid 대신 CEL - softmax
'''


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
    
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
    dropout_prob = trial.suggest_float('dropout_prob', 1e-2, 1e-1)
    lr = trial.suggest_float('lr', 1e-2, 1e-1)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BinaryClassificationModel(input_size = 4, hidden_size=hidden_size, dropout_prob=dropout_prob)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    torchtrial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)
    torchtrial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)

    history = torchtrial.run(epochs=20)

    result = torchtrial.evaluate(data_key=torchbearer.TEST_DATA)
    test_accuracy = result['test_binary_acc']
    return test_accuracy


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    storage_dir = '../day47/train/spaceship/optuna'
    study = optuna.create_study(pruner=MedianPruner(), direction='maximize', study_name="fc_tuning", storage=f'sqlite:///{storage_dir}/fc.db')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial: ')
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    model = BinaryClassificationModel(input_size = 4, hidden_size=trial.params['hidden_size'], dropout_prob=trial.params['dropout_prob'])
    optimizer = optim.Adam(model.parameters(), lr=trial.params['lr'])
    criterion = nn.BCELoss()
    best_try = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)

    model_path = os.path.join(pathFolder, "fc_best_model.pth")
    torch.save(model, model_path)

    print("Model saved to: ", model_path)