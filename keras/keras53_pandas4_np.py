import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col = 0, header = 0)
#                                                 ㄴ 안해주면 index도 데이터 취급한다.
print(df)

print(df.shape)     # (150, 5)
print(df.info())

# Pandas -> numpy 변환
# 방법 1. to_numpy()
aaa = df.to_numpy()
print(aaa)
print(type(aaa))

# 방법 2. values
bbb = df.values
print(aaa)
print(type(aaa))

np.save('../data/npy/iris_sklearn.npy', arr = aaa)

# 과제
# 판다스의 loc iloc에 대해 정리
# 1. iloc: 행의 