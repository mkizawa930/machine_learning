import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def to_lowercase(names):
    new_names = {}
    for name in names:
        new_names[name] = str.lower(name)

    return new_names

df = pd.read_csv('./N225.csv')
new_columns = to_lowercase(df.columns)
df.rename(new_columns, axis=1, inplace=True)
df.set_index('date', inplace=True)

df = df[df.index > '2010-01-01']
df['close'].plot()

df['return'] = np.log(df['close'] / df['close'].shift(1))

df['y'] = df['return']

df = df[['y']]

# add lag features
for lag in range(1, 12):
    df[f'y_lag_{lag}'] = df['y'].shift(lag)

window_size = 14
df[f'SMA_{window_size}'] = df['y'].shift(1).rolling(window_size).mean()

df.dropna(inplace=True)

# 二値分類問題
df['target'] = np.where(df['y'] > 0, 1, 0)

train_valid, test = train_test_split(df, test_size=0.2, shuffle=False)
train, valid = train_test_split(train_valid, test_size=0.2, shuffle=False)

target = 'target'
features = list(filter(lambda name: name not in [target, 'y'], df.columns))
print(features)
train_X, train_y = train[features], train[target]
test_X, test_y = test[features], test[target]


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
model = SVC()
model.fit(train_X, train_y)

test_y_pred = model.predict(test_X)

acc = accuracy_score(test_y.values, test_y_pred)
acc

prc = precision_score(test_y.values, test_y_pred)
prc

rcc = recall_score(test_y.values, test_y_pred)
rcc

test['y'].cumsum().plot()
(test_y_pred * test['y']).cumsum().plot()


