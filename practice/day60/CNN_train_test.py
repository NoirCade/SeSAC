import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import pickle
import gzip


class TextCNNDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data.iloc[idx]
        dat = np.array(dat)
        dat = dat.astype('float32')
        label = self.labels.iloc[idx]

        if self.transform:
            dat = self.transform(dat)

        return dat, label


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (6, 127), padding=(2,0))
        self.conv2 = nn.Conv2d(64, 128, (9, 1), padding=(4,0))
        self.conv3 = nn.Conv2d(128, 256, (12, 1), padding=(6,0))
        self.fc1 = nn.Linear(1792, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))
        # print('conv 1 ', x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))
        # print('conv 2 ', x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), (3, 1))
        # print('conv 3 ', x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as t:
        for data, labels in dataloader:
            # data = data.unsqueeze(1)
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(1)
        
    average_loss = total_loss / len(dataloader)
    print(f"Training loss: {average_loss}")
    return average_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Validation", unit="batch") as t:
            for data, labels in dataloader:
                # data = data.unsqueeze(1)
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += (predicted == labels).sum().item()

                t.update(1)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Validation Loss: {average_loss}, Accuracy: {accuracy}")
    return average_loss, accuracy


def cnn_trainer(X, y, epochs, batch_size, model_path=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    best_accuracy = 0.0
    best_model = None
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        mean = 0.5
        std = 0.5

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        
        train_dataset = TextCNNDataset(X_train, y_train, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        val_dataset = TextCNNDataset(X_val, y_val, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # 모델 인스턴스화 및 옵티마이저 설정
        model = TextCNN()
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # GPU 사용 설정 (가능한 경우)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 20)

        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model.state_dict()
            torch.save(best_model, f'best_model.pth')
            print('Best model saved!')
    print('Best model Accuracy == ', best_accuracy)


def cnn_tester(X, y, model_path):
    mean = 0.5
    std = 0.5

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    
    test_dataset = TextCNNDataset(X, y, transform=transform)
    test_dataloader = DataLoader(test_dataset)
    model = TextCNN()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0
    correct_predictions = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc="Test", unit="batch") as t:
            for data, labels in test_dataloader:
                # data = data.unsqueeze(1)
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                t.update(1)

    average_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / len(test_dataloader.dataset)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    with open('predicted.txt', 'w') as f:
        for label in predicted_labels:
            f.write(f'{label}, ')

    print('Test loss == ', average_loss, ' Test Accuracy == ', accuracy, ' F1 Score == ', f1, ' Prescision == ', precision, ' Recall == ', recall)
    print('Confusion Matrix: ')
    print(conf_matrix)



if __name__ == '__main__':
    # pickle 파일 로드
    with gzip.open('Xtrain.pickle', 'rb') as f:
        Xtrain = pickle.load(f)

    data = Xtrain

    with gzip.open('ytrain.pickle', 'rb') as f:
        ytrain = pickle.load(f)
    
    label = ytrain

    cnn_trainer(data, label, 100, 16)

    with gzip.open('Xtest.pickle', 'rb') as f:
        Xtest = pickle.load(f)

    data = Xtest

    with gzip.open('ytest.pickle', 'rb') as f:
        ytest = pickle.load(f)
    
    label = ytest
    model = 'best_model.pth'
    # cnn_tester(data, label, model)
