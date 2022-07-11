import pandas as pd
from pyarrow import feather
from pyarrow import parquet
from datetime import datetime
from itertools import chain
import gc
from dask import dataframe as dd

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
time_dict = {datetime(k.year, k.month, 1):v for k, v in zip(time_range, range(0, 14))}
data['pay_idx'] = data['S_2'].apply(lambda x: time_dict[datetime(x.year, x.month, 1)])


#%% customer id 처리 : shorten
data['id'] = data['customer_ID'].apply(lambda x: x[-10:])
id_dict = {shorten: origin for origin, shorten in zip(data['customer_ID'].unique(), data['id'].unique())}


#%% Get numeric data
cat_cols = ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
cont_cols = ['id', 'pay_idx', 'target']
remove_cols = ['customer_ID', 'S_2']

num_cols = [x for x in data.columns if (x not in cat_cols) and (x not in cont_cols) and ((x not in remove_cols))]


num_data = data.loc[:, ['id'] + num_cols]
cat_data = data.loc[:, ['id'] + cat_cols]


del data
gc.collect()



#%%


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
    연속형 변수에 aggregation 기본 제공 항목에 대한 groupby 진행.
    :param data: groupby 진행할 dataframe
    :param groupby_col_name: groupby 기준 컬럼 명
    :param agg_list: aggregation 수행할 항목 list
    :return:
    """
    dask_df =
    result = data.groupby(groupby_col_name).agg(agg_list)
    result.columns = ['_'.join(x) for x in result.columns]
    result = result.reset_index()
    return result



num_basic_summ_feats = get_basic_summary_feats(num_data, 'id')

num_basic_summ_feats.insert(0, 'customer_ID', num_basic_summ_feats['id'].apply(lambda x: id_dict[x]))
num_basic_summ_feats = num_basic_summ_feats.drop(['id'], axis=1)

num_basic_summ_feats.to_parquet('./SH/middle_output/numeric_basic_summary.parquet',
                                engine='fastparquet')

