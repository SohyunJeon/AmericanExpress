"""
Description: 기본 코딩 가이드
Version: 1.0
Date: 2022-06-28
"""


#%% Packages
import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import confusion_matrix
from datetime import date, timedelta



#%% setting
os.chdir('D:/kaggle_2022_default_prediction/data')
SEED = 2022
random.seed(SEED)



#%% Function
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()
    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)
    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    return 0.5 * (g + d)







#%% import Data
# trains = pd.read_csv('train_data.csv')
# tests = pd.read_csv('test_data.csv')
# filename = 'test_data.csv'
# chunksize = 1000*100
# for tests in pd.read_csv(filename, chunksize=chunksize):

labels = pd.read_csv('train_labels.csv')
submission = pd.read_csv('sample_submission.csv')
trains = pd.read_feather('train_data.ftr') # 5531451, 192
tests = pd.read_feather('test_data.ftr') # 11363762, 190




#%% EDA
# sample_trains = trains.iloc[0:10000,].to_csv('sample_trains.csv')
sample_trains = trains.iloc[0:1000,]
len(trains['customer_ID'].unique()) # 458,913
len(tests['customer_ID'].unique()) # 924,621
labels['target'].sum() / len(labels['target']) # 25.89 %



#%% Pre-processing
all_cols = [i for i in list(trains.columns) if i not in ['customer_ID','S_2','target']]
cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
num_features = [col for col in all_cols if col not in cat_features]
# trains = trains.merge(labels, on='customer_ID', how='left')




#%% Feature engineering
# sample test
# sample_trains_groups = pd.DataFrame()
# sample_trains_groups['customer_ID'] = sample_trains['customer_ID'].unique()
# sample_y = labels['target'][0:83]
#
# feat_name = [i + '_avg' for i in num_features]
# for i, j in enumerate(num_features):
#     temp = sample_trains.groupby('customer_ID')[j].mean()
#     summary_dataframe = pd.DataFrame()
#     summary_dataframe['customer_ID'] = temp.index
#     summary_dataframe[feat_name[i]] = temp.values
#     sample_trains_groups = pd.merge(sample_trains_groups, summary_dataframe, on='customer_ID', how='left')


# train
trains_groups = pd.DataFrame()
trains_groups['customer_ID'] = trains['customer_ID'].unique()
train_y = labels['target']

feat_name = [i + '_avg' for i in num_features]
for i, j in enumerate(num_features):
    temp = trains.groupby('customer_ID')[j].mean()
    summary_dataframe = pd.DataFrame()
    summary_dataframe['customer_ID'] = temp.index
    summary_dataframe[feat_name[i]] = temp.values
    trains_groups = pd.merge(trains_groups, summary_dataframe, on='customer_ID', how='left')

feat_name = [i + '_max' for i in num_features]
for i, j in enumerate(num_features):
    temp = trains.groupby('customer_ID')[j].max()
    summary_dataframe = pd.DataFrame()
    summary_dataframe['customer_ID'] = temp.index
    summary_dataframe[feat_name[i]] = temp.values
    trains_groups = pd.merge(trains_groups, summary_dataframe, on='customer_ID', how='left')

feat_name = [i + '_min' for i in num_features]
for i, j in enumerate(num_features):
    temp = trains.groupby('customer_ID')[j].min()
    summary_dataframe = pd.DataFrame()
    summary_dataframe['customer_ID'] = temp.index
    summary_dataframe[feat_name[i]] = temp.values
    trains_groups = pd.merge(trains_groups, summary_dataframe, on='customer_ID', how='left')




#%% Modeling
X_train, X_valid, y_train, y_valid = train_test_split(trains_groups, train_y, test_size=0.3, random_state=SEED,  stratify=train_y)
X_train = X_train.loc[:, X_train.columns != 'customer_ID']
X_valid = X_valid.loc[:, X_valid.columns != 'customer_ID']

model = XGBClassifier(base_score=0.7, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
              gamma=0, gpu_id=-1, importance_type='gain',
              interaction_constraints='',
              learning_rate=0.03,
              n_estimators=100,
              max_delta_step=0,
              max_depth=6,
              min_child_weight=1, missing=np.nan,
              monotone_constraints='()',
              n_jobs=16,
              num_parallel_tree=1, objective='binary:logistic', random_state=SEED,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1, verbosity=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)


# evaluation
# np.unique(y_pred, return_counts=True)
# np.unique(y_valid, return_counts=True)
y_true = pd.DataFrame({'target' : y_valid['y_valid']})
y_pred = pd.DataFrame({'prediction' : y_pred['prediction']})



print(amex_metric(y_valid, y_pred))
confusion_matrix(y_valid, y_pred)





#%% Submission
# feature
tests_groups = pd.DataFrame()
tests_groups['customer_ID'] = tests['customer_ID'].unique()
train_y = labels['target']

feat_name = [i + '_avg' for i in num_features]
for i, j in enumerate(num_features):
    temp = tests.groupby('customer_ID')[j].mean()
    summary_dataframe = pd.DataFrame()
    summary_dataframe['customer_ID'] = temp.index
    summary_dataframe[feat_name[i]] = temp.values
    tests_groups = pd.merge(tests_groups, summary_dataframe, on='customer_ID', how='left')

feat_name = [i + '_max' for i in num_features]
for i, j in enumerate(num_features):
    temp = tests.groupby('customer_ID')[j].max()
    summary_dataframe = pd.DataFrame()
    summary_dataframe['customer_ID'] = temp.index
    summary_dataframe[feat_name[i]] = temp.values
    tests_groups = pd.merge(tests_groups, summary_dataframe, on='customer_ID', how='left')

feat_name = [i + '_min' for i in num_features]
for i, j in enumerate(num_features):
    temp = tests.groupby('customer_ID')[j].min()
    summary_dataframe = pd.DataFrame()
    summary_dataframe['customer_ID'] = temp.index
    summary_dataframe[feat_name[i]] = temp.values
    tests_groups = pd.merge(tests_groups, summary_dataframe, on='customer_ID', how='left')


# prediction
tests_groups = tests_groups.loc[:, tests_groups.columns != 'customer_ID']
test_pred = model.predict(tests_groups)
np.unique(test_pred, return_counts=True)
submission['prediction'] = test_pred


# file
# sum(tests['customer_ID'].unique() ==  submission['customer_ID']) #924621
today = str(date.today())
submission.to_csv('submisson_'+today+'.csv', index=False)






