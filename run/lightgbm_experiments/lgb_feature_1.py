#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: gjwei
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)
import glob
from feature_gen.utils import read_data
import gc


param = {'num_leaves': 111,
         'min_data_in_leaf': 149,
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 133,
         "verbosity": -1}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

train = read_data('./input/train.csv')
test = read_data('./input/test.csv')
# print(train.head())
target = train['target']

train['outlier'] = 0
train.loc[train['target'] < -30, 'outlier'] = 1
print(train['outlier'].value_counts())

del train['target']
gc.collect()

for file_path in glob.glob("./input/features/*.csv"):
    print('merge feature {}'.format(file_path))
    feature_df = pd.read_csv(file_path)
    train = pd.merge(train, feature_df, on='card_id', how='left')
    test = pd.merge(test, feature_df, on='card_id', how='left')

    del feature_df
    gc.collect()

features = [c for c in train.columns if c not in ['card_id', 'first_active_month', 'outliers']]
categorical_feats = ['feature_2', 'feature_3']

# print(train.head())

oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['outliers'].values)):
    print("fold n / {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature=categorical_feats
                           )
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature=categorical_feats
                           )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target) ** 0.5))

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('./run/lightgbm_experiments/lgbm_importances.png')

sub_df = pd.DataFrame({"card_id": test["card_id"].values})
sub_df["target"] = predictions

now = datetime.datetime.strftime(datetime.datetime.now(), '%b-%d-%y %H:%M')
sub_df.to_csv("results/submission_{}.csv".format(now), index=False)