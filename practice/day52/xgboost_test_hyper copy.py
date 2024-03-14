import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


pathFolder = "./train/spaceship/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "XTrain.pkl"
yTrainName = "yTrain.pkl"
with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

print(type(X),type(y))

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# XGBoost 분류기 인스턴스 생성
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 하이퍼파라미터 검색 공간 정의
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}

# 그리드 탐색 객체 초기화
grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', verbose=10)

# 그리드 탐색 수행
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters found: ", grid_search.best_params_)

# 최적 모델로 예측
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# 예측 정확도 평가
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

best_model.save_model('best_model.xgb')