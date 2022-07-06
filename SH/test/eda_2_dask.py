import pandas as pd
from pyarrow import feather
from itertools import chain
import dask.dataframe as dd
from catboost import CatBoostClassifier

#%%
import dask
data = feather.read_feather('./data/train_data.ftr')
ds_data = dask.dataframe.from_pandas(data, npartitions=5)


#%%

data = feather.read_feather('./data/train_data.ftr')


cols_S = [col for col in data.columns if ('S_' in col) and (col != 'S_2')] # 소비
cols_D = [col for col in data.columns if 'D_' in col] # 연체
cols_P = [col for col in data.columns if 'P_' in col] # 지불
cols_B = [col for col in data.columns if 'B_' in col] # 잔고
cols_R = [col for col in data.columns if 'R_' in col] # 리스크

cols_set = [cols_S, cols_D, cols_P, cols_B, cols_R]


#%%  Groupby test


target_df = data.groupby('customer_ID').first()['target']






#%% Na check
def get_na_info(df:pd.DataFrame) -> (pd.Series, pd.Series):
    obs_cnt = len(df)
    na_cnt = df.isna().sum()
    na_ratio = na_cnt/obs_cnt
    return na_cnt, na_ratio

na_ratio_limit = 0.5


remove_cols_list = []
for cols in cols_set:
    na_cnt, na_ratio = get_na_info(data.loc[:, cols])
    rem_cols = na_ratio[na_ratio > na_ratio_limit].index.tolist()
    remove_cols_list += rem_cols


#%% class 그룹별 min, max, avg summary 생성
"""
차이가 있는 변수만 통계량을 사용
"""
pos_df = data.loc[data['target'] == 1, :]
neg_df = data.loc[data['target'] == 0, :]

def make_basic_summ_features(df: pd.DataFrame, groupby_col_name:str) -> pd.DataFrame:
    """
    :param df: groupby 컬럼을 포함한 dataframe
    :param groupby_col_name: groupby 적용할 컬럼명 (ex:id)
    :return:
    """
    feats = ['avg', 'min', 'max']
    result = pd.DataFrame()
    for feat in feats:
        if feat == 'avg':
            feat_df = df.groupby(groupby_col_name).mean().add_suffix('_'+feat)
            result = pd.concat([result, feat_df], axis=1)

        elif feat == 'min':
            feat_df = df.groupby(groupby_col_name).min().add_suffix('_'+feat)
            result = pd.concat([result, feat_df], axis=1)
        elif feat == 'max':
            feat_df = df.groupby(groupby_col_name).max().add_suffix('_'+feat)
            result = pd.concat([result, feat_df], axis=1)
        else:
            pass
    return result


summ_dict = {}
for cols in cols_set:
    neg_summ = make_basic_summ_features(neg_df.loc[:, ['customer_ID']+cols], 'customer_ID')
    pos_summ = make_basic_summ_features(pos_df.loc[:, ['customer_ID'] + cols], 'customer_ID')
    col_name = cols[0].split('_')[0]
    summ_dict[col_name+'_neg_summ'] = neg_summ
    summ_dict[col_name + '_pos_summ'] = pos_summ

#%% 분포 비교




#%% 범주형
cat_cols = ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
cat_df = data.loc[:, ['customer_ID', 'S_2', 'target'] + cat_cols]

cat_df['B_31'].value_counts()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.countplot(cat_df['D_63'], data=cat_df,  hue='target')
plt.show()


#%% 숫자형
num_cols = [col for col in chain(*cols_set) if not col in cat_cols]
num_cols = [col for col in num_cols if not col in remove_cols_list]
num_df = data.loc[:, ['customer_ID'] + num_cols]

num_summ_df = make_basic_summ_features(num_df, 'customer_ID')


#%% feature selection with catboost
target_df = data.groupby('customer_ID').first()['target']
cat_df_agg = data.loc[:, ['customer_ID'] + cat_cols].groupby('customer_ID').last()
cat_df_agg = cat_df_agg.astype(str)


total_df = pd.concat([target_df, num_summ_df, cat_df_agg], axis=1)
total_df = total_df.reset_index().drop('customer_ID', axis=1)

train_df = total_df.loc[:int(len(total_df)*0.7), :]
valid_df = total_df.loc[int(len(total_df)*0.7):, :]

train_X = train_df.drop('target', axis=1)
train_y = train_df['target']

valid_X = valid_df.drop('target', axis=1)
valid_y = valid_df['target']



model = CatBoostClassifier()
model.fit(train_X, train_y, cat_cols)


pred = model.predict_proba(valid_X)
pred_prob = pd.DataFrame({'prediction': pred[:, 1]})
# pred_prob = pd.DataFrame({'prediction': pred})


#%% eval
from modules.Evaluation import amex_metric


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


y_true = pd.DataFrame(valid_y).reset_index(drop=True)
score = amex_metric(y_true, pred_prob)

#%% 메모리 정리
import gc
del target_df, cat_df_agg, total_df, num_summ_df
gc.collect()

#%% Test
test = data = feather.read_feather('./data/test_data.ftr')

# target
test_ids = test['customer_ID'].unique()

# numeric
test_num_df = test.loc[:, ['customer_ID'] + num_cols]
test_num_summ_df = make_basic_summ_features(test_num_df, 'customer_ID')

# category
test_cat_df_agg = test.loc[:, ['customer_ID'] + cat_cols].groupby('customer_ID').last()
test_cat_df_agg = test_cat_df_agg.astype(str)

# merge
test_total_df = pd.concat([test_num_summ_df, test_cat_df_agg], axis=1)
test_total_df = test_total_df.reset_index()
test_value = test_total_df.drop('customer_ID', axis=1)
# predict
test_pred = model.predict_proba(test_value)
test_result = pd.DataFrame({'customer_ID': test_total_df['customer_ID'].values,
                            'prediction': test_pred[:, 1]})

test_result.to_csv('./output/submission_220705.csv', index=False)