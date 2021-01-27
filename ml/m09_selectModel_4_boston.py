from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()
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
        print(name, '은 없는 놈!')
        # continue


import sklearn
print(sklearn.__version__)      # 0.23.2

# Tensorflow
# 0.9183033302819957

# ARDRegression 의 정답률:  0.7217518756242229
# AdaBoostRegressor 의 정답률:  0.8248438711324639
# BaggingRegressor 의 정답률:  0.8839600184708266
# BayesianRidge 의 정답률:  0.7248735659759427
# CCA 의 정답률:  0.6111074076621716
# DecisionTreeRegressor 의 정답률:  0.7431660083221863
# DummyRegressor 의 정답률:  -0.027399755882764554
# ElasticNet 의 정답률:  0.6992890673117649
# ElasticNetCV 의 정답률:  0.6928590110451693
# ExtraTreeRegressor 의 정답률:  0.7081314932211044
# ExtraTreesRegressor 의 정답률:  0.917387953171472
# GammaRegressor 의 정답률:  -0.027399755882764998
# GaussianProcessRegressor 의 정답률:  -6.444308207970055
# GeneralizedLinearRegressor 의 정답률:  0.7109777801443053
# GradientBoostingRegressor 의 정답률:  0.8986977205598934
# HistGradientBoostingRegressor 의 정답률:  0.9054872486044216
# HuberRegressor 의 정답률:  0.6331565367287234
# KNeighborsRegressor 의 정답률:  0.7179317681541104
# KernelRidge 의 정답률:  0.7206674772338744
# Lars 의 정답률:  0.6044979538575129
# LarsCV 의 정답률:  0.6245441567453622
# Lasso 의 정답률:  0.6975599845192237
# LassoCV 의 정답률:  0.7160019147833843
# LassoLars 의 정답률:  -0.027399755882764554
# LassoLarsCV 의 정답률:  0.7332455671272153
# LassoLarsIC 의 정답률:  0.7335737571322061
# LinearRegression 의 정답률:  0.7269774874059812
# LinearSVR 의 정답률:  0.6723138121202316
# MLPRegressor 의 정답률:  0.5450856492469079
# NuSVR 의 정답률:  0.3304615746705978
# OrthogonalMatchingPursuit 의 정답률:  0.5502083400032283
# OrthogonalMatchingPursuitCV 의 정답률:  0.6744668318674397
# PLSCanonical 의 정답률:  -2.9440510414419876
# PLSRegression 의 정답률:  0.7044061261193344
# PassiveAggressiveRegressor 의 정답률:  0.3415674929535638
# PoissonRegressor 의 정답률:  0.7834550543588651
# RANSACRegressor 의 정답률:  0.3160808314757515
# RandomForestRegressor 의 정답률:  0.9019230039689486
# Ridge 의 정답률:  0.7306005477249637
# RidgeCV 의 정답률:  0.7284172670280171
# SGDRegressor 의 정답률:  -2.135775648403042e+27
# SVR 의 정답률:  0.31922597192734314
# TheilSenRegressor 의 정답률:  0.7174848896194864
# TransformedTargetRegressor 의 정답률:  0.7269774874059812
# TweedieRegressor 의 정답률:  0.7109777801443053