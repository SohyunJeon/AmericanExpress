"""


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

#%%

data = feather.read_feather('./SH/data/train_data.ftr')

#%%  지불일자와 y -> 지불 일자는 동일하지 않음

temp1 = data.loc[:, ['customer_ID', 'S_2']]
temp1.group

del temp1
gc.collect()

#%% numeric summary 변수의 상관관계
temp1 = pd.read_parquet(os.path.join(conf.feats_store_dir, 'numeric_basic_summary.parquet'))


def get_na_info(df:pd.DataFrame) -> (pd.Series, pd.Series):
    obs_cnt = len(df)
    na_cnt = df.isna().sum()
    na_ratio = na_cnt/obs_cnt
    return na_cnt, na_ratio

na_ratio_limit = 0.4

temp1_na_cnt, temp1_na_ratio = get_na_info(temp1)

drop_cols = temp1_na_ratio[temp1_na_ratio > na_ratio_limit].index.tolist()

temp1_na_drop = temp1.drop(drop_cols + ['customer_ID'], axis=1)

for val in ['mean', 'min', 'max', 'last', 'first']:
    temp1_val = temp1_na_drop.loc[:, [x for x in temp1_na_drop.columns if val in x]]
    temp1_corr = temp1_val.corr()
    mask = np.triu(np.ones_like(temp1_val.corr(), dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(temp1_corr, mask=mask, vmax=.3, linewidth=.5)
    plt.title(val)
    plt.tight_layout()
    plt.show()



temp1_corr.iloc[0, :]