import pandas as pd
import numpy as np

# 1. Load Data
score = pd.read_csv('../data/csv/data-01-test-score.csv')
wine = pd.read_csv('../data/csv/winequality-white.csv', sep = ';', index_col = None, header = 0)

x = wine.iloc[:, :-1].values
y = wine.iloc[:, -1].values

# 방법 1
# https://ponyozzang.tistory.com/291

count_data = wine.groupby('quality')['quality'].count()
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64

# 방법 2
unique, counts = np.unique(y, return_counts = True)
print(np.asarray((unique, counts)).T)

# 방법 3
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(y)
plt.show()
# y값: 3, 4, 5, 6, 7, 8, 9