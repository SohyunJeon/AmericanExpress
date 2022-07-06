#%%
import pandas as pd
import numpy as np
from datetime import datetime
# from dataprep.eda import create_report
import gc
import os
import pickle
import findspark
findspark.init()


# import plotly
# import plotly.graph_objs as go

# import matplotlib.pyplot as plt


import pyspark
import databricks.koalas as ks
import pyspark.pandas as ps

from pyarrow import feather

#%% koalas test
dates = pd.date_range('20220323', periods=6)
pdf = pd.DataFrame(np.random.randn(6,4), index=dates)

kdf = ks.from_pandas(pdf)




#%%
#
# train_raw = pd.read_csv('./data/train_data.csv', nrows=10)
train_ft = pd.read_feather('./data/train_data.ftr')
# test_ft = pd.read_feather('./data/test_data.ftr')

# train_ft_32 = pd.read_feather('./data/train_data_f32.ftr')

train_df = ps.read_csv('./data/train_data.csv')

train_data = feather.read_feather('./data/train_data.ftr')






#%% 변수 특징별 데이터 추출
cols_S = ['customer_ID', 'target'] + [col for col in train_ft.columns if 'S_' in col] # 소비
cols_D = ['customer_ID', 'target', 'S_2'] + [col for col in train_ft.columns if 'D_' in col] # 연체
cols_P = ['customer_ID', 'target', 'S_2'] + [col for col in train_ft.columns if 'P_' in col] # 지불
cols_B = ['customer_ID', 'target', 'S_2'] + [col for col in train_ft.columns if 'B_' in col] # 잔고
cols_R = ['customer_ID', 'target', 'S_2'] + [col for col in train_ft.columns if 'R_' in col] # 리스크


tr_S = train_ft.loc[:,  cols_S]
tr_D = train_ft.loc[:,  cols_D]
tr_P = train_ft.loc[:,  cols_P]
tr_B = train_ft.loc[:,  cols_B]
tr_R = train_ft.loc[:,  cols_R]

#%%

id_list = set(train_ft['customer_ID'])
len(id_list)


#%% dataprep 레포트 생성


def make_eda_report(data:pd.DataFrame, data_cnt:int=1000, report_name:str='test'):
    report_data = data.loc[:data_cnt, :]
    report = create_report(report_data)
    report.save(f'./middle_output/{report_name}')
    print(f'{report_name} save done.')


make_eda_report(tr_S, report_name='sample_S_report')
make_eda_report(tr_D, report_name='sample_D_report')
make_eda_report(tr_P, report_name='sample_P_report')
make_eda_report(tr_B, report_name='sample_B_report')
make_eda_report(tr_R, report_name='sample_R_report')

# 메모리 제거
del tr_S, tr_D, tr_P, tr_B, tr_R
gc.collect(generation=2)


#%%
train_data = train_ft.copy()

# id dict 생성
total_ids = train_data['customer_ID'].unique().tolist()
id_dict = {k: v for k, v in zip(total_ids, range(len(total_ids)))}
# with open('./middle_output/train_id_dict.pkl', 'wb') as f:
#     pickle.dump(id_dict, f)



train_data.insert(0, 'ID', train_data['customer_ID'].apply(lambda x: id_dict[x]))
train_data.insert(0, 'dt', train_data.insert(0, 'dt', train_data['S_2'].apply(lambda x: str(x.year)+'_'+str(x.month))))
train_data.drop('customer_ID', axis=1, inplace=True)
train_data.drop('S_2', axis=1, inplace=True)



neg_ids = train_data.loc[train_data['target']==0, 'ID'].unique().tolist()
pos_ids = train_data.loc[train_data['target']==1, 'ID'].unique().tolist()



#%% 날짜별(달) 값의 변화
neg_df = train_data.loc[train_data['ID'].isin(neg_ids), :]
pos_df = train_data.loc[train_data['ID'].isin(pos_ids), :]

neg_mean = neg_df.groupby(['dt']).mean()
pos_mean = pos_df.groupby(['dt']).mean()

neg_median = neg_df.groupby(['dt']).median()
pos_median = pos_df.groupby(['dt']).median()

neg_max = neg_df.groupby(['dt']).max()
pos_max = pos_df.groupby(['dt']).max()

neg_min = neg_df.groupby(['dt']).min()
pos_min = pos_df.groupby(['dt']).min()



def chart_per_month_compare_label(data_1:pd.DataFrame, data_2:pd.DataFrame,
                                  dir_name: str, col_name: str, labels: list=['neg', 'pos']):
    plt.figure()
    plt.plot(data_1.index.values, data_1[col_name], label=labels[0])
    plt.plot(data_2.index.values, data_2[col_name], label=labels[1])
    plt.title(col_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'{col_name}.jpg'))
    plt.clf()



for col in neg_mean.columns:
    chart_per_month_compare_label(neg_mean, pos_mean, './middle_chart/month_mean', col)

for col in neg_median.columns:
    if col in ['ID', 'target']:
        continue
    chart_per_month_compare_label(neg_median, pos_median, './middle_chart/month_median', col)

for col in neg_max.columns:
    if col in ['ID', 'target']:
        continue
    chart_per_month_compare_label(neg_max, pos_max, './middle_chart/month_max', col)


for col in neg_min.columns:
    if col in ['ID', 'target']:
        continue
    chart_per_month_compare_label(neg_min, pos_min, './middle_chart/month_min', col)