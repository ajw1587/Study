import pandas as pd

train = pd.read_csv('F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/train.csv')
test = pd.read_csv('F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/test.csv')
submission = pd.read_csv('F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/sample_submission.csv')

print(train.isna().sum())