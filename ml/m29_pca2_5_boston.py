# 차원축소 PCA
# 주성분 분석

import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)                     # (506, 13) (506,)

# pca = PCA(n_components = 9)
# x2 = pca.fit_transform(x)
# print(x2.shape)                             

# pca_EVR = pca.explained_variance_ratio_     # 압축시킨 9개의 Column 중요성 비율
# print(pca_EVR)
# print(sum(pca_EVR))

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum: ', cumsum)       # 밑에 설명
# cumsum:  [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
#  1.        ]

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95', cumsum >= 0.95)
print('d: ', d)
# cumsum >= 0.95 [False  True  True  True  True  True  True  True  True  True  True  True  True]

# 그래프
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

##########################################################################################
# np.cumsum: 각 원소들의 누적 합을 표시함. 각 row와 column의 구분은 없어지고, 순서대로 sum을 함.

# print np.cumsum(a, dtype = float)
# --> 결과 : 
# [  1.   3.   6.  10.  15.  21.]
# --> 설명 : 결과 값의 변수 type을 설정하면서 누적 sum을 함.


# print np.cumsum(a, axis = 0)
# --> 결과 : 
# [[1 2 3]
#  [5 7 9]]
# --> 설명: axis = 0은 같은 column 끼리의 누적 합을 함.

# print np.cumsum(a, axis = 1)
# --> 결과 : 
# [[ 1  3  6]
#  [ 4  9 15]]
# --> 설명 : axis = 1은 같은 row끼리의 누적 합을 함.
##########################################################################################
