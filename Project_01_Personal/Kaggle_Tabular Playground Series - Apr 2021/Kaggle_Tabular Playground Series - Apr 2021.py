# https://www.kaggle.com/salilchoubey/tps-apr-eda-very-baseline-model
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as py
import plotly.express as go

train = pd.read_csv('F:\Personal Project\Kaggle\Tabular Playground Series - Apr 2021/train.csv')
test = pd.read_csv('F:\Personal Project\Kaggle\Tabular Playground Series - Apr 2021/test.csv')
submission = pd.read_csv('F:\Personal Project\Kaggle\Tabular Playground Series - Apr 2021/sample_submission.csv')

train.info()
'''
is_na = pd.DataFrame(train.isna().sum(),columns=['Number'])
is_na['Percent'] = is_na['Number']/100000
is_na = is_na.sort_values('Percent',ascending=False)

print("The number of unique values in the column Cabin are:", train['Cabin'].nunique())

train['Cabin_alpha'] = train['Cabin'].str.replace('[^a-zA-Z]', '')
test['Cabin_alpha'] = test['Cabin'].str.replace('[^a-zA-Z]', '')

print(train['Cabin_alpha'].value_counts())
print(train['Cabin_alpha'].isna().sum())

train['Cabin_alpha'].fillna('NA',inplace=True)
test['Cabin_alpha'].fillna('NA',inplace=True)

train.groupby('Cabin_alpha')['Pclass'].value_counts()
train.groupby('Cabin_alpha')['Survived'].mean().sort_values()
go.bar(train.groupby('Cabin_alpha')['Survived'].mean().sort_values())

print("The number of unique values in the column Ticket are:", train['Ticket'].nunique())

train['Ticket_alpha'] = train['Ticket'].str.replace('[^a-zA-Z]', '')
test['Ticket_alpha'] = test['Ticket'].str.replace('[^a-zA-Z]', '')

train['Ticket_num'] = train['Ticket'].str.replace('[^0-9]', '')
test['Ticket_num'] = test['Ticket'].str.replace('[^0-9]', '')

train['Ticket_alpha'].fillna('NA',inplace=True)
train['Ticket_alpha'].replace({'':'NA'},inplace=True)
test['Ticket_alpha'].fillna('NA',inplace=True)
test['Ticket_alpha'].replace({'':'NA'},inplace=True)

train['Ticket_num'].fillna('0',inplace=True)
train['Ticket_num'].replace({'':'0'},inplace=True)
test['Ticket_num'].fillna('0',inplace=True)
test['Ticket_num'].replace({'':'0'},inplace=True)

train.head()

train['Ticket_num'] = train['Ticket_num'].astype(int)
test['Ticket_num'] = test['Ticket_num'].astype(int)
train[['Ticket_alpha','Ticket_num']].dtypes

train.drop(columns=['Cabin','Ticket'],inplace=True)
test.drop(columns=['Cabin','Ticket'],inplace=True)

# EDA
x = pd.DataFrame(train.groupby('Pclass')['Cabin_alpha'].value_counts())
x.columns = ['Counts']
x = x.reset_index()

x['Percentage']=train.groupby('Pclass')['Cabin_alpha'].value_counts().groupby(level=0).apply(lambda 
        x:100 * x/float(x.sum())).values

go.bar(x,x='Pclass',y='Percentage',color='Cabin_alpha')

fig = go.bar(train['Pclass'].value_counts())
fig.update_layout(title='Number of passengers in each Class',xaxis_title='Pclass',yaxis_title='Number of passengers')
fig.show()

y = pd.DataFrame(train.groupby('Pclass')['Sex'].value_counts())
y.columns = ['Counts']
y = y.reset_index()
y['Percentage']=train.groupby('Pclass')['Sex'].value_counts().groupby(level=0).apply(lambda 
        x:100 * x/float(x.sum())).values
y['Pclass'] = y['Pclass'].astype('category')

go.bar(y,x='Pclass',y='Percentage',color='Sex')

z = pd.DataFrame(train.groupby(['Pclass','Sex'])['Survived'].mean()).reset_index()
go.bar(z,x='Pclass',y='Survived',color='Sex',barmode='group')

sns.boxplot(x=train['Survived'],y=train['Age'])

age = train[['Age','Survived','Sex']].dropna()
bins = [0,15, 50, 200]
labels = ['Child', 'Adult', 'Old']
age['age_band'] = pd.cut(age.Age, bins, labels = labels,include_lowest = True)
go.bar(age.groupby(['age_band'])['Survived'].mean())

for i in ['Embarked','Sex','Pclass']:
    print('Value counts for column',i,'are:')
    print(train[i].value_counts())
    print('-'*50)

train['Name'].head(10)

train.drop(columns = 'Name',inplace=True)
train.head()

is_na = pd.DataFrame(train.isna().sum(),columns=['Number'])
is_na['Percent'] = is_na['Number']/100000
is_na = is_na.sort_values('Percent',ascending=False)

train['Age'].hist()

print('Age mean;',train['Age'].mean())
print('Age median:',train['Age'].median())
print('Age skew:',train['Age'].skew())

print(train.groupby('Survived')['Age'].mean())
print('-'*50)
print(train.groupby('Survived')['Age'].median())

print(train.groupby('Pclass')['Age'].mean())
print('-'*50)
print(train.groupby('Pclass')['Age'].median())

for df in [train,test]:
    for i in [1,2,3]:
        a = df[df['Age'].isna()][['Pclass','Age']]
        ind = list(a[a['Pclass']==i].index)
        df.loc[ind,'Age'] = df[df['Pclass']==i]['Age'].mean()


train['Age'].skew()
for df in [train,test]:
    df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

train['Fare'].hist()
print('Mean:',train['Fare'].mean())
print('Median:',train['Fare'].median())

train.groupby('Pclass')['Fare'].mean()
train.groupby('Pclass')['Fare'].median()

for df in [train,test]:
    for i in [1,2,3]:
        a = df[df['Fare'].isna()][['Pclass','Fare']]
        ind = list(a[a['Pclass']==i].index)
        df.loc[ind,'Fare'] = df[df['Pclass']==i]['Fare'].median()
is_na = pd.DataFrame(train.isna().sum(),columns=['Number'])
is_na['Percent'] = is_na['Number']/100000
is_na = is_na.sort_values('Percent',ascending=False)

train.head()
train.groupby('Cabin_alpha')['Fare'].describe()
sns.heatmap(train.corr(),annot=True,cmap='crest')

train['Family_members'] = train['SibSp'] + train['Parch']
test['Family_members'] = test['SibSp'] + test['Parch']
train['Alone'] = 0
test['Alone'] = 0
train.loc[train['Family_members']==0,'Alone'] = 1
test.loc[test['Family_members']==0,'Alone'] = 1
    
sns.heatmap(train.corr(),annot=True,cmap='crest')
train.columns

fig = go.bar(train.drop(columns=['PassengerId']).corr().loc['Survived','Pclass':'Alone'])
fig.update_layout(title='Correlation of features with Surival',xaxis_title='Features',yaxis_title='Correlation value')
fig.show()

train['Fare'] = np.log(train['Fare'])
test['Fare'] = np.log(test['Fare'])
train['Fare'].skew()

print(set(train.columns) - set(test.columns))
print(set(test.columns) - set(train.columns))

test.drop(columns='Name',inplace=True)
train.groupby('Cabin_alpha')['Fare'].describe().T

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train[train.dtypes[train.dtypes == object].index].nunique()

for i in ['Cabin_alpha','Ticket_alpha']:
    train[i] = lb.fit_transform(train[i])
    test[i] = lb.transform(test[i])
train.head()

train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.drop(columns=['Sex_male'],inplace=True)
test.drop(columns=['Sex_male'],inplace=True)
import statsmodels.api as sm
def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features
features = forward_selection(train.drop(columns=['PassengerId','Survived']),train['Survived']) 

id_col = test['PassengerId']
for df in [train,test]:
    df.drop(columns='PassengerId',inplace=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
rfc = RandomForestClassifier(max_depth=10, min_samples_leaf=10, min_samples_split=100)
X = train.drop(columns='Survived')
y = train['Survived']
X.head()

lr.fit(X.drop(columns=['Ticket_alpha','Ticket_num','Family_members']),y)
cv_means = cross_val_score(lr,X.drop(columns=['Ticket_alpha','Ticket_num','Family_members']),y)
np.mean(cv_means)

submission = pd.DataFrame({'PassengerID':id_col,'Survived':lr.predict(test.drop(columns=['Ticket_alpha','Ticket_num','Family_members']))})
submission = submission.set_index('PassengerID')
submission.to_csv('F:\Personal Project\Kaggle\Tabular Playground Series - Apr 2021/result_submission.csv')
'''