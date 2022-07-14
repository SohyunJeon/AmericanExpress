import pandas as pd


def get_summary(df):
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2','target']]
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68', 'D_63', 'D_64', 'D_66']
    b_features = [col for col in list(df.columns) if col.startswith('B_') and col not in cat_features]
    p_features = [col for col in list(df.columns) if col.startswith('P_') and col not in cat_features]
    num_features = [col for col in all_cols if col not in cat_features and col not in b_features and col not in p_features]

    # B_ 변수
    train_b_agg = df.groupby("customer_ID")[b_features].agg(['last', 'nunique', 'count'])
    # P_ 변수
    train_p_agg = df.groupby("customer_ID")[p_features].agg(['mean', 'std', 'min', 'last','var','median','count'])
    # category 변수
    train_cat_agg = df.groupby("customer_ID")[cat_features].agg(['last', 'nunique'])
    # B_, P_, category를 제외한 numeric 변수
    train_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last', 'var', 'median', 'count'])

    result = pd.concat([train_num_agg, train_b_agg, train_p_agg, train_cat_agg], axis=1)
    return result

