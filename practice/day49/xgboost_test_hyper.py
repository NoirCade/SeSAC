import numpy as np
import os
import pickle
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


pathFolder = "../day47/train/spaceship/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "XTrain.pkl"
yTrainName = "yTrain.pkl"
with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

print(type(X),type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

def objective(trial):
    param = {
        # "device": 'cuda',
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        "num_boost_around": trial.suggest_int("num_boost_around", 100, 1000),
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000, timeout=600)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial: ')
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_bst = xgb.train(trial.params, dtrain)

    model_path = os.path.join(pathFolder, "xgb_best_model.pth")
    with open(model_path, "wb") as f:
        pickle.dump(best_bst, f)

    print("Model saved to: ", model_path)

    opt_path = os.path.join(pathFolder, "optimization_history.png")
    param_path = os.path.join(pathFolder, "param_importances.png")
    plot_optimization_history(study, save_file=opt_path)
    plot_param_importances(study, save_file=param_path)


# # 그리드 탐색 객체 초기화
# grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', verbose=10)

# # 그리드 탐색 수행
# grid_search.fit(X_train, y_train)

# # 최적 하이퍼파라미터 출력
# print("Best parameters found: ", grid_search.best_params_)

# # 최적 모델로 예측
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(X_test)

# # 예측 정확도 평가
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# best_model.save_model('best_model.xgb')