from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

allAlgorithms = all_estimators(type_filter = 'regressor')
# sklearn에 들어있는 classifier 모델들이 들어있다.

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', r2_score(y_test, y_pred))
    except:
        # print(name, '은 없는 놈!')
        continue 


import sklearn
print(sklearn.__version__)      # 0.23.2

# Tensorflow
# 0.5095695895801424

# ARDRegression 의 정답률:  0.49884657895071893
# AdaBoostRegressor 의 정답률:  0.47033083380331175
# BaggingRegressor 의 정답률:  0.4565811043017928
# BayesianRidge 의 정답률:  0.498195294090212
# CCA 의 정답률:  0.40684858195972107
# DecisionTreeRegressor 의 정답률:  0.13224576277967393
# DummyRegressor 의 정답률:  -0.0020572917785151024
# ElasticNet 의 정답률:  0.006280808832653917
# ElasticNetCV 의 정답률:  0.4503042150288753
# ExtraTreeRegressor 의 정답률:  -0.05804637777873811
# ExtraTreesRegressor 의 정답률:  0.4803436003930359
# GammaRegressor 의 정답률:  0.003996819655354367
# GaussianProcessRegressor 의 정답률:  -14.231591146656491
# GeneralizedLinearRegressor 의 정답률:  0.0041418802042252345
# GradientBoostingRegressor 의 정답률:  0.48446698503101093
# HistGradientBoostingRegressor 의 정답률:  0.4531747521142405
# HuberRegressor 의 정답률:  0.4803393265686049
# KNeighborsRegressor 의 정답률:  0.4583895962258292
# KernelRidge 의 정답률:  -3.4833370185393493
# Lars 의 정답률:  -0.7612620070535188
# LarsCV 의 정답률:  0.49722637185634255
# Lasso 의 정답률:  0.3337638148085368
# LassoCV 의 정답률:  0.498190206774737
# LassoLars 의 정답률:  0.35911832707748426
# LassoLarsCV 의 정답률:  0.4979932905460819
# LassoLarsIC 의 정답률:  0.49713833418827535
# LinearRegression 의 정답률:  0.5034051724671986
# LinearSVR 의 정답률:  -0.39237319922546954     
# MLPRegressor 의 정답률:  -2.917397476923105
# NuSVR 의 정답률:  0.1638068193331651
# OrthogonalMatchingPursuit 의 정답률:  0.25506432717267025
# OrthogonalMatchingPursuitCV 의 정답률:  0.4951150921983971
# PLSCanonical 의 정답률:  -0.5530684368929146
# PLSRegression 의 정답률:  0.5077759812192139
# PassiveAggressiveRegressor 의 정답률:  0.4891376225563052 
# PoissonRegressor 의 정답률:  0.3456306069999251
# RANSACRegressor 의 정답률:  0.40580723502784843
# RadiusNeighborsRegressor 의 정답률:  -0.0020572917785151024
# RandomForestRegressor 의 정답률:  0.46826548770842025
# Ridge 의 정답률:  0.42957028992465474
# RidgeCV 의 정답률:  0.49909594114392664
# SGDRegressor 의 정답률:  0.42433442437218116
# SVR 의 정답률:  0.1449245914430527
# TheilSenRegressor 의 정답률:  0.4844028022419864
# TransformedTargetRegressor 의 정답률:  0.5034051724671986
# TweedieRegressor 의 정답률:  0.0041418802042252345     