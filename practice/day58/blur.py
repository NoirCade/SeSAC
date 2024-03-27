from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].squeeze(0)  # DataLoader에서 배치 처리를 위해 차원 축소
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label



class HateSpeechClassifier(nn.Module):
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("skt/kobert-base-v1")
        self.classifier = nn.Linear(768, 2)  # KoBERT의 hidden size: 768, 클래스 수: 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # CLS 토큰의 출력을 사용
        logits = self.classifier(cls_output)
        return logits


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as t:
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
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
            for input_ids, attention_mask, labels in dataloader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += (predicted == labels).sum().item()

                t.update(1)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Validation Loss: {average_loss}, Accuracy: {accuracy}")
    return average_loss, accuracy


if __name__ == '__main__':
    # KoBERT 모델과 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    model = AutoModel.from_pretrained("skt/kobert-base-v1")

    data = pd.read_csv('./clean_or_dirty_words.csv').drop(columns='Unnamed: 0')
    X = data['문장']
    y = data['clean']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    test_dataset = HateSpeechDataset(X_val, y_val, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # 모델 인스턴스화 및 옵티마이저 설정
    model = HateSpeechClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # GPU 사용 설정 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    # 학습 파라미터
    epochs = 10
    best_accuracy = 0.0
    best_model = None
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 20)

        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, test_dataloader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pth')
            print('Best model saved!')

