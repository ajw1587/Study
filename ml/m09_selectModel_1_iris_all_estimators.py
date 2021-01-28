from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
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
# 0.9666666388511658

# AdaBoostClassifier 의 정답률:  0.9
# BaggingClassifier 의 정답률:  0.8666666666666667
# BernoulliNB 의 정답률:  0.26666666666666666
# CalibratedClassifierCV 의 정답률:  0.9
# CategoricalNB 의 정답률:  0.9
# CheckingClassifier 의 정답률:  0.3
# ComplementNB 의 정답률:  0.7333333333333333
# DecisionTreeClassifier 의 정답률:  0.9
# DummyClassifier 의 정답률:  0.3
# ExtraTreeClassifier 의 정답률:  0.9
# ExtraTreesClassifier 의 정답률:  0.8666666666666667
# GaussianNB 의 정답률:  0.8666666666666667
# GaussianProcessClassifier 의 정답률:  0.9
# GradientBoostingClassifier 의 정답률:  0.8666666666666667
# HistGradientBoostingClassifier 의 정답률:  0.8333333333333334
# KNeighborsClassifier 의 정답률:  0.9333333333333333
# LabelPropagation 의 정답률:  0.9
# LabelSpreading 의 정답률:  0.9
# LinearDiscriminantAnalysis 의 정답률:  0.9333333333333333
# LinearSVC 의 정답률:  0.9
# LogisticRegression 의 정답률:  0.8666666666666667
# LogisticRegressionCV 의 정답률:  0.9333333333333333
# MLPClassifier 의 정답률:  0.9333333333333333
# MultinomialNB 의 정답률:  0.7666666666666667
# NearestCentroid 의 정답률:  0.8666666666666667
# NuSVC 의 정답률:  0.9
# PassiveAggressiveClassifier 의 정답률:  0.9333333333333333
# Perceptron 의 정답률:  0.5666666666666667
# QuadraticDiscriminantAnalysis 의 정답률:  0.9333333333333333
# RadiusNeighborsClassifier 의 정답률:  0.9
# RandomForestClassifier 의 정답률:  0.8666666666666667
# RidgeClassifier 의 정답률:  0.7333333333333333
# RidgeClassifierCV 의 정답률:  0.7333333333333333
# SGDClassifier 의 정답률:  0.8333333333333334
# SVC 의 정답률:  0.9