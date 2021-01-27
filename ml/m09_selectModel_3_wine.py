from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

allAlgorithms = all_estimators(type_filter = 'classifier')
# sklearn에 들어있는 classifier 모델들이 들어있다.

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', accuracy_score(y_test, y_pred))
    except:
        print(name, '은 없는 놈!')
        # continue 그냥 넘기고 싶을때


import sklearn
print(sklearn.__version__)      # 0.23.2

# Tensorflow
# 1.0

# AdaBoostClassifier 의 정답률:  0.9166666666666666
# BaggingClassifier 의 정답률:  0.9166666666666666
# BernoulliNB 의 정답률:  0.3888888888888889
# CalibratedClassifierCV 의 정답률:  0.8611111111111112
# CheckingClassifier 의 정답률:  0.3055555555555556
# ComplementNB 의 정답률:  0.6388888888888888
# DecisionTreeClassifier 의 정답률:  0.9166666666666666
# DummyClassifier 의 정답률:  0.3611111111111111
# ExtraTreeClassifier 의 정답률:  0.9166666666666666
# ExtraTreesClassifier 의 정답률:  1.0
# GaussianNB 의 정답률:  0.9722222222222222
# GaussianProcessClassifier 의 정답률:  0.4722222222222222
# GradientBoostingClassifier 의 정답률:  0.9444444444444444
# HistGradientBoostingClassifier 의 정답률:  0.9722222222222222
# KNeighborsClassifier 의 정답률:  0.7222222222222222
# LabelPropagation 의 정답률:  0.3888888888888889
# LabelSpreading 의 정답률:  0.3888888888888889
# LinearDiscriminantAnalysis 의 정답률:  0.9722222222222222
# LinearSVC 의 정답률:  0.9166666666666666
# LogisticRegression 의 정답률:  0.9166666666666666
# LogisticRegressionCV 의 정답률:  0.9444444444444444
# MLPClassifier 의 정답률:  0.3888888888888889
# MultinomialNB 의 정답률:  0.7777777777777778
# NearestCentroid 의 정답률:  0.7222222222222222
# NuSVC 의 정답률:  0.8611111111111112
# PassiveAggressiveClassifier 의 정답률:  0.4444444444444444
# Perceptron 의 정답률:  0.6111111111111112
# QuadraticDiscriminantAnalysis 의 정답률:  0.9722222222222222
# RandomForestClassifier 의 정답률:  0.9722222222222222
# RidgeClassifier 의 정답률:  1.0
# RidgeClassifierCV 의 정답률:  1.0
# SGDClassifier 의 정답률:  0.5833333333333334
# SVC 의 정답률:  0.6388888888888888