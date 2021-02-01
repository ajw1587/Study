# Computer_Vision
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 y -> 0~9, 다중분류
file_path = '../data/csv/Computer_Vision/data/train.csv'
dataset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949', index_col = 0)
# print(type(dataset))        # <class 'numpy.ndarray'>
# print(dataset.shape)        # (2048, 786)

x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# print(type(x))              # <class 'numpy.ndarray'>
# print(x.shape)              # (2048, 785)
# print(type(y))              # <class 'numpy.ndarray'>
# print(y.shape)              # (2048,)

# 2. PCA
pca = PCA()
pca.fit(x)