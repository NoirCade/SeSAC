import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.models as models

pathFolder = "./train/spaceship/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "XTrain.pkl"
yTrainName = "yTrain.pkl"
with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

print(type(X),type(y))

