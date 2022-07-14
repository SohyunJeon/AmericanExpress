from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# preprocessing
def get_summary_and_diff_preprocessing(train):
    features = train.drop(['customer_ID', 'S_2','target'], axis=1).columns.to_list()
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68', 'D_63', 'D_64', 'D_66']
    num_features = [col for col in features if col not in cat_features]
    return cat_features, num_features


# features
def get_summary_and_diff(train, train_labels):
    cat_features, num_features = get_summary_and_diff_preprocessing(train)

    # summary
    ## numeric
    train_num_agg = train.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)
    ## category
    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)

    ## Transform float64 columns to float32
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    ## Transform int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)

    # diff
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(train.groupby(['customer_ID'])):
        ## Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        ## Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    ## Concatenate
    df1 = np.concatenate(df1, axis=0)
    ## Transform to dataframe
    df1 = pd.DataFrame(df1, columns=[col + '_diff1' for col in df[num_features].columns])
    ## Add customer id
    df1['customer_ID'] = customer_ids

    # merge
    df = train_num_agg.merge(train_cat_agg, how='inner', on='customer_ID').merge(df1, how='inner',on='customer_ID').merge(train_labels, how='inner', on='customer_ID')

    return df




