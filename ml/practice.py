from sklearn.datasets import load_boston
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
kfold = KFold(n_splits = 5, shuffle = True)

# model
Algorithm = all_estimators(type_filter = 'regressor')

for (name, algo) in Algorithm:
    try:
        model = algo()
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print(name, '의 r2_score: \n', scores)
    except:
        continue