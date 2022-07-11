"""
- 연속형 변수 기본 summary
- datetime 변수 관련 feature 생성

"""

#%%
import pandas as pd
import numpy as np
from pyarrow import feather
from pyarrow import parquet
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import chain
import gc
from dask import dataframe as dd
import pickle

#%%

data = feather.read_feather('./SH/data/train_data.ftr')
test_data = feather.read_feather('./SH/data/test_data.ftr')

cols_S = [col for col in data.columns if ('S_' in col) and (col != 'S_2')] # 소비
cols_D = [col for col in data.columns if 'D_' in col] # 연체
cols_P = [col for col in data.columns if 'P_' in col] # 지불
cols_B = [col for col in data.columns if 'B_' in col] # 잔고
cols_R = [col for col in data.columns if 'R_' in col] # 리스크

cols_set = [cols_S, cols_D, cols_P, cols_B, cols_R]


del test_data, cols_S, cols_D, cols_P, cols_B, cols_R, cols_set
gc.collect()

#%% date 처리 : 2017.03 : 0 ~ 2018.3 : 12
time_range = pd.date_range('2017-03', '2018-04', freq='M')
time_dict = {datetime(k.year, k.month, 1): v for k, v in zip(time_range, range(0, 14))}
data['pay_idx'] = data['S_2'].apply(lambda x: time_dict[datetime(x.year, x.month, 1)])


#%% customer id 처리 : shorten
data['id'] = data['customer_ID'].apply(lambda x: x[-10:])
id_dict = {shorten: origin for origin, shorten in zip(data['customer_ID'].unique(), data['id'].unique())}

with open('./SH/middle_output/id_dict.pkl', 'wb') as f:
    pickle.dump(id_dict, f)

#%% Get numeric data
cat_cols = ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
cont_cols = ['id', 'pay_idx', 'target']
remove_cols = ['customer_ID', 'S_2']

num_cols = [x for x in data.columns if (x not in cat_cols) and (x not in cont_cols) and ((x not in remove_cols))]


num_data = data.loc[:, ['id'] + num_cols]
cat_data = data.loc[:, ['id'] + cat_cols]
date_data = data.loc[:, ['id', 'pay_idx', 'target']]


del data
gc.collect()

feather.write_feather(num_data, './SH/data/train_numeric_data.ftr')
feather.write_feather(cat_data, './SH/data/train_categorical_data.ftr')
feather.write_feather(date_data, './SH/data/train_date_data.ftr')


#%% Numeric summary


def get_basic_summary_feats(data:pd.DataFrame, groupby_col_name:str, agg_list:list=['mean', 'std', 'min', 'max', 'last', 'first']):
    """
    연속형 변수에 aggregation 기본 제공 항목에 대한 groupby 진행.
    :param data: groupby 진행할 dataframe
    :param groupby_col_name: groupby 기준 컬럼 명
    :param agg_list: aggregation 수행할 항목 list
    :return:
    """
    result = data.groupby(groupby_col_name).agg(agg_list)
    result.columns = ['_'.join(x) for x in result.columns]
    result = result.reset_index()
    return result



def get_basic_summary_feats_dask(data:pd.DataFrame, groupby_col_name:str, agg_list:list=['mean', 'std', 'min', 'max', 'last', 'first']):
    """
    (Dask 사용) 연속형 변수에 aggregation 기본 제공 항목에 대한 groupby 진행.
    :param data: groupby 진행할 dataframe
    :param groupby_col_name: groupby 기준 컬럼 명
    :param agg_list: aggregation 수행할 항목 list
    :return:
    """
    dask_df = dd.from_pandas(data, chunksize=5)
    result = dask_df.groupby(groupby_col_name).agg(agg_list)
    result.columns = ['_'.join(x) for x in result.columns]
    result = result.reset_index()

    return result



num_basic_summ_feats = get_basic_summary_feats(num_data, 'id')

num_basic_summ_feats.insert(0, 'customer_ID', num_basic_summ_feats['id'].apply(lambda x: id_dict[x]))
num_basic_summ_feats = num_basic_summ_feats.drop(['id'], axis=1)

num_basic_summ_feats.to_parquet('./SH/middle_output/numeric_basic_summary.parquet',
                                engine='fastparquet')


## dask 저장이 너무 오래 걸림
# dask_num_basic_summ_feats = get_basic_summary_feats_dask(num_data, 'id')
#
# dask_num_basic_summ_feats.to_parquet('./SH/middle_output/dask_numeric_basic_summary.parquet',
#                                      engine='fastparquet')

del num_basic_summ_feats
gc.collect()


#%% Date features


def period(x: np.array) -> np.array:
    return x.max() - x.min()


def get_month_index(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """
    datetime 컬럼에서 month 기준, 시간 순서대로 int로 변환 (첫달 -> 0)
    :param df:
    :param date_col_name: datetime 컬럼명
    :return:
    """
    start_month = datetime.strftime(df[date_col_name].min(), '%Y-%m')
    end_month = datetime.strftime(df[date_col_name].max() + relativedelta(months=1), '%Y-%m')
    time_range = pd.date_range(start_month, end_month, freq='M', inclusive='right')
    time_dict = {datetime(k.year, k.month, 1): v for k, v in zip(time_range, range(0, len(time_range)))}
    df['paymonth_idx'] = df[date_col_name].apply(lambda x: time_dict[datetime(x.year, x.month, 1)])
    df = df.drop(date_col_name, axis=1)
    return df


def period(x: np.array) -> np.array:
        return x.max() - x.min()


def get_payment_basic_feats(df: pd.DataFrame, date_col_name: str, groupby_col_name: str) -> pd.DataFrame:
    """
    :param df: groupby 진행할 dataframe
    :param date_col_name: datetime 컬럼 명
    :param groupby_col_name: groupby 기준 컬럼 명
    :return: groupby 결과
    """
    df = df.loc[:, [groupby_col_name, date_col_name]]
    # change payment month to int
    df = get_month_index(df, date_col_name)

    # agg: first, last, period
    result = df.groupby(groupby_col_name).agg(['first', 'last', period])
    result.columns = ['_'.join(x) for x in result.columns]
    result = result.reset_index()

    return result


train_payment_feats = get_payment_basic_feats(data, date_col_name='S_2', groupby_col_name='customer_ID')

train_payment_feats.to_parquet('./SH/middle_output/paydate_basic.parquet',
                                engine='fastparquet')


