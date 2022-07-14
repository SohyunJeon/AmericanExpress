"""
1. https://www.kaggle.com/code/btbpanda/fast-metric-and-py-boost-baseline

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
