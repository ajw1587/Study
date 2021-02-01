# 차원축소 PCA
# 주성분 분석

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)                     # (442, 10) (442,)

pca = PCA(n_components = 9)
x2 = pca.fit_transform(x)
print(x2.shape)                             # (442, 10) -> (442, 7)

pca_EVR = pca.explained_variance_ratio_     # 압축시킨 7개의 Column 중요성 비율
print(pca_EVR)
print(sum(pca_EVR))
# 7개: 0.9479436357350414
# 8개: 0.9913119559917797
# 9개: 0.9991439470098977