#%%

import pandas as pd
import os

from SH.config import conf

#%%
num_basic = pd.read_parquet(os.path.join(conf.feats_store_dir, 'numeric_basic_summary.parquet'))
num_added = pd.read_parquet(os.path.join(conf.feats_store_dir, 'numeric_basic_summary_added.parquet'))
paydate_basic = pd.read_parquet(os.path.join(conf.feats_store_dir, 'paydate_basic.parquet'))
cat_ohe = pd.read_parquet(os.path.join(conf.feats_store_dir, 'cat_last_ohe.parquet'))