import pandas as pd

from sklearn.model_selection import train_test_split
from woe.sklearn import WoeTransformer


pd.set_option('display.max_columns', 999)

data = pd.read_csv('UCI_Credit_Card.csv')
WoeTransformer.create_new_config('woe_config.csv', data)
WoeTransformer.mark_woe_features('woe_config.csv',
                                 ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3'])

X = data.drop('target', axis=1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
                                       X, y, test_size=0.33, random_state=42)

woe_trans = WoeTransformer(
                config_filepath='woe_config.csv',
                save_woe_pickle_filepath='woe_weights.pkl',
                min_sample_weight=0.2)

woe_trans.fit(X_train, y_train)
X_train_woe = woe_trans.transform(X_train)
X_test_woe = woe_trans.transform(X_test)

woe_trans2 = WoeTransformer(
                config_filepath='woe_config.csv',
                load_woe_pickle_filepath='woe_weights.pkl')
X_train_woe = woe_trans.transform(X_train)
X_test_woe = woe_trans.transform(X_test)
