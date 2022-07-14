"""
1. https://www.kaggle.com/code/btbpanda/fast-metric-and-py-boost-baseline
https://www.kaggle.com/code/susnato/amex-data-preprocesing-feature-engineering

"""


#%%
import pandas as pd
import numpy as np
from pyarrow import feather
import os
from pyarrow import parquet
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import chain
import gc
from dask import dataframe as dd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from SH.config import conf
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#%%

data = feather.read_feather('./SH/data/train_data.ftr')

#%% 4
cat_cols = ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
cat_df = data.loc[:, ['customer_ID'] + cat_cols]

groupby_col_name = 'customer_ID'
cat_agg = cat_df.groupby(groupby_col_name).last()

cat_agg.columns = ['_'.join([x, 'last']) for x in cat_cols]

cat_last = cat_agg.astype(object)
ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')
ohe.fit(cat_last)
cat_ohe = ohe.transform(cat_last)

df = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names_out())
df.insert(0, groupby_col_name, cat_agg.index.values)


df.to_parquet('./SH/middle_output/cat_last_ohe.parquet',
                                engine='fastparquet')


def get_cat_last_ohe(
        data: pd.DataFrame,
        groupby_col_name:str='customer_ID'
) -> pd.DataFrame:
    """

    :param data:
    :param groupby_col_name: groupby 기줕 컬럼명
    :return: one-hot-encoding 결과
    """
    cat_cols = ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    cat_df = data.loc[:, [groupby_col_name] + cat_cols]
    cat_agg = cat_df.groupby(groupby_col_name).last()
    cat_agg.columns = ['_'.join([x, 'last']) for x in cat_cols]
    cat_last = cat_agg.astype(object)

    ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')
    ohe.fit(cat_last)
    cat_ohe = ohe.transform(cat_last)

    result = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names_out())
    result.insert(0, groupby_col_name, cat_agg.index.values)

    return result


del cat_ohe
gc.collect()

#%% 7

bin_cols = ['B_31', 'D_87']
cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

groupby_col_name = 'customer_ID'
df = data.loc[:, [groupby_col_name, 'S_2']]

result = df.groupby(groupby_col_name).count()
result.columns = ['S_2'+'_count']

statements = df.groupby(groupby_col_name)['S_2'].agg('diff').dt.days
statements = statements.fillna(0)
statements_df = pd.DataFrame(statements.values, columns=['days_btw_pay'])
statements_df.insert(0, groupby_col_name, df[groupby_col_name])

days_btw_pay = statements_df.groupby(groupby_col_name).agg(['mean', 'std', 'max', 'min'])
days_btw_pay.columns = ["_".join(x) for x in days_btw_pay.columns]

days_fl = df.groupby(groupby_col_name).agg(['first', 'last'])
days_fl.columns = ["_".join(x) for x in days_fl.columns]


days_fl['S_2_first_diff'] = (days_fl['S_2_first'].min() - days_fl['S_2_first']).dt.days
days_fl['S_2_last_diff'] = (days_fl['S_2_last'].max() - days_fl['S_2_last']).dt.days

for d_col in ['first', 'last']:
    days_fl[f'S_2_{d_col}_dd'] = days_fl[f'S_2_{d_col}'].dt.day
    days_fl[f'S_2_{d_col}_mm'] = days_fl[f'S_2_{d_col}'].dt.month
    days_fl[f'S_2_{d_col}_yy'] = days_fl[f'S_2_{d_col}'].dt.year

days_fl = days_fl.drop('S_2_first', axis=1)

def last2(series):
    return series.values[-2] if len(series.values)>=2 else -127

def last3(series):
    return series.values[-3] if len(series.values)>=3 else -127

last_more = df.groupby(groupby_col_name).agg([last2, last3])
last_more.columns = ["_".join(x) for x in last_more.columns]

statements_last_more = statements_df.groupby(groupby_col_name).agg([last2, last3])
statements_last_more.columns = ["_".join(x) for x in statements_last_more.columns]


total_result = pd.concat([days_btw_pay, statements_last_more,  days_fl, last_more ], axis=1)

total_result = total_result.reset_index()




total_result.to_csv('./SH/middle_output/paydate_adv_1.csv')


feather.write_feather(total_result, './SH/middle_output/paydate_adv_1.ftr')



def get_paydate_features(data: pd.DataFrame) -> pd.DataFrame:
    groupby_col_name = 'customer_ID'
    df = data.loc[:, [groupby_col_name, 'S_2']]

    # 고객당 지불 일자별 기간에 대한 summary
    statements = df.groupby(groupby_col_name)['S_2'].agg('diff').dt.days
    statements = statements.fillna(0)
    statements_df = pd.DataFrame(statements.values, columns=['days_btw_pay'])
    statements_df.insert(0, groupby_col_name, df[groupby_col_name])

    days_btw_pay = statements_df.groupby(groupby_col_name).agg(['mean', 'std', 'max', 'min'])
    days_btw_pay.columns = ["_".join(x) for x in days_btw_pay.columns]

    # 고객별 지불 일자의 max(고객 천체 처음값) - 처음값 /  min(고객 전체 마지막값) - 마지막값
    days_fl = df.groupby(groupby_col_name).agg(['first', 'last'])
    days_fl.columns = ["_".join(x) for x in days_fl.columns]

    days_fl['S_2_first_diff'] = (days_fl['S_2_first'].min() - days_fl['S_2_first']).dt.days
    days_fl['S_2_last_diff'] = (days_fl['S_2_last'].max() - days_fl['S_2_last']).dt.days

    # 마지막 일자의 datetime 분해
    for d_col in ['first', 'last']:
        days_fl[f'S_2_{d_col}_dd'] = days_fl[f'S_2_{d_col}'].dt.day
        days_fl[f'S_2_{d_col}_mm'] = days_fl[f'S_2_{d_col}'].dt.month
        days_fl[f'S_2_{d_col}_yy'] = days_fl[f'S_2_{d_col}'].dt.year

    days_fl = days_fl.drop('S_2_first', axis=1)

    # 지불 일자의 한달 전, 두달전 값
    last_more = df.groupby(groupby_col_name).agg([last2, last3])
    last_more.columns = ["_".join(x) for x in last_more.columns]

    statements_last_more = statements_df.groupby(groupby_col_name).agg([last2, last3])
    statements_last_more.columns = ["_".join(x) for x in statements_last_more.columns]

    # 취합
    total_result = pd.concat([days_btw_pay, statements_last_more, days_fl, last_more], axis=1).reset_index()

    return total_result
