from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
# 0.9912280440330505

# AdaBoostClassifier 의 정답률:  0.956140350877193
# BaggingClassifier 의 정답률:  0.9298245614035088
# BernoulliNB 의 정답률:  0.6578947368421053
# CalibratedClassifierCV 의 정답률:  0.9122807017543859
# CheckingClassifier 의 정답률:  0.34210526315789475
# ComplementNB 의 정답률:  0.8859649122807017
# DecisionTreeClassifier 의 정답률:  0.9298245614035088
# DummyClassifier 의 정답률:  0.5526315789473685
# ExtraTreeClassifier 의 정답률:  0.9035087719298246
# ExtraTreesClassifier 의 정답률:  0.956140350877193
# GaussianNB 의 정답률:  0.9298245614035088
# GaussianProcessClassifier 의 정답률:  0.9035087719298246
# GradientBoostingClassifier 의 정답률:  0.956140350877193
# HistGradientBoostingClassifier 의 정답률:  0.956140350877193
# KNeighborsClassifier 의 정답률:  0.9385964912280702
# LabelPropagation 의 정답률:  0.38596491228070173
# LabelSpreading 의 정답률:  0.38596491228070173
# LinearDiscriminantAnalysis 의 정답률:  0.956140350877193
# LinearSVC 의 정답률:  0.9035087719298246
# LogisticRegression 의 정답률:  0.9385964912280702
# LogisticRegressionCV 의 정답률:  0.9385964912280702
# MLPClassifier 의 정답률:  0.9210526315789473
# MultinomialNB 의 정답률:  0.8859649122807017
# NearestCentroid 의 정답률:  0.8771929824561403
# NuSVC 의 정답률:  0.8508771929824561
# PassiveAggressiveClassifier 의 정답률:  0.9210526315789473
# Perceptron 의 정답률:  0.9122807017543859
# QuadraticDiscriminantAnalysis 의 정답률:  0.9649122807017544
# RandomForestClassifier 의 정답률:  0.9473684210526315
# RidgeClassifier 의 정답률:  0.956140350877193
# RidgeClassifierCV 의 정답률:  0.956140350877193
# SGDClassifier 의 정답률:  0.9298245614035088
# SVC 의 정답률:  0.9122807017543859