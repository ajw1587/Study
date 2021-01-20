import numpy as np
import pandas as pd

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
result = np.array(range(0, 7776))
result = result.reshape(7776, 1)
# print(result)
# print(result.shape)
# x1 = x[:, 0:1]
# x2 = x[:, 1:2]
# print(x1.shape)
# print(x2.shape)
# y = np.vstack((x1, x2))
# print(y)
# result = np.hstack((result, y))
# print(result)
# print(result.shape)

result = pd.DataFrame(result[:,1:])
# print(result.shape)
print(type(result))
result.iloc

df = pd.read_csv('../Sunlight/sample_submission.csv')
# df = df.loc[df.]
# print(df.index)
# print(df.columns)
# print(df.id)
