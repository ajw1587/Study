import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col = 0, header = 0)
#                                                 ㄴ 안해주면 index도 데이터 취급한다.
print(df)