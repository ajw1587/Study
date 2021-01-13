import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names',
#            'DESCR', 'feature_names', 'filename'])
print(dataset.values())
print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica']

# x = dataset.data
# y = dataset.target
x = dataset['data']
y = dataset['target']

print(x)
print(y)
print(x.shape)              # (150, 4)
print(y.shape)              # (150,)
print(type(x), type(y))     # <class 'numpy.ndarray>

# df = pd.DataFrame(x, columns=dataset['feature_names'])
df = pd.DataFrame(x, columns=dataset.feature_names)
print(df)
print('\n')
print(df.shape)
print('\n')
print(df.columns)
print('\n')
print(df.index)
print('\n')

print(df.head())        # = df[:5]
print(df.tail())        # = df[-5:]
print('\n')
print(df.info())        # non-null: 결측치가 없다.
print('\n')
print(df.describe())    # 데이터의 Max, Min, Mean, Std 등...
print('\n')

# Columns 명 수정
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.columns)
print('\n')
print(df.info())
print('\n')
print(df.describe())
print('\n')

# y Column 추가
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head())

print(df.shape)     # (150, 5)
print(df.columns)
print(df.index)
print(df.tail())

print(df.info())
print(df.isnull())  # null값 판단
print(df.isnull().sum())
print(df.describe())
print(df['Target'].value_counts())

# 상관계수 히트맵
print(df.corr())
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data = df.corr(), square = True, annot = True, cbar = True)
# plt.show()


# 도수 분포도 (histogram)
plt.figure(figsize = (10, 6))       # 크기 잡아주기

plt.subplot(2, 2, 1)
plt.hist(x = 'sepal_length', data = df)
plt.title('sepal_length')

plt.subplot(2, 2, 2)
plt.hist(x = 'sepal_width', data = df)
plt.title('sepal_width')

plt.subplot(2, 2, 3)
plt.hist(x = 'petal_length', data = df)
plt.title('petal_length')

plt.subplot(2, 2, 4)
plt.hist(x = 'petal_width', data = df)
plt.title('petal_width')

plt.show()