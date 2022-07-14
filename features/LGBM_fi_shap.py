import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

def get_summary_and_diff_preprocessing(train):
    features = train.drop(['customer_ID', 'S_2','target'], axis=1).columns.to_list()
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68', 'D_63', 'D_64', 'D_66']
    num_features = [col for col in features if col not in cat_features]
    return cat_features, num_features

def get_LGBM_FeatureImportance_ShapValue(df):
    # custom_ID별 기간(day 기준), 시작 month 정보를 계산
    agg_train = pd.DataFrame()
    df['S_2'] = pd.to_datetime(df['S_2'])
    agg_train['S_2_min'] = df[['S_2','customer_ID']].groupby('customer_ID')['S_2'].first()
    agg_train['S_2_max'] = df[['S_2','customer_ID']].groupby('customer_ID')['S_2'].last()
    agg_train['S_2_period'] = (agg_train['S_2_max'] - agg_train['S_2_min']).dt.days
    agg_train['S_2_start_month'] = agg_train['S_2_min'].apply(lambda x: x.strftime('%m'))

    # custom_ID별 mean, std값 계산
    def agg_feat_eng(df):
        cat_features, num_features = get_summary_and_diff_preprocessing(df)
        #num_features = [col for col in df.columns if is_numeric_dtype(df[col]) and col != 'target']
        agg_feature_names = [f'{feat}_mean' for feat in num_features] + [f'{feat}_std' for feat in num_features]
        num_feats_agg = df.groupby('customer_ID')[num_features].agg(['mean','std'])
        num_feats_agg.columns = agg_feature_names
        return num_feats_agg
    train_agg = agg_feat_eng(df)
    # custom_ID별로 마지막 값(행)만 가져옴
    train_df = df.groupby('customer_ID').tail(1).set_index('customer_ID')

    # merge
    train_df = pd.concat([train_df, train_agg, agg_train[['S_2_period','S_2_start_month']]], axis=1)

    # object -> category type
    object_cols = [col for col in train_df.columns if str(train_df[col].dtype) == 'object']
    train_df[object_cols] = train_df[object_cols].astype('category')

    # split train, validation
    train_cols = [col for col in train_df.columns if col not in ['customer_ID','target','S_2']]
    X_train, X_val, y_train, y_val = train_test_split(train_df[train_cols], train_df['target'], test_size=0.2, random_state=42)

    # lightGBM modeling
    dtrain = lgb.Dataset(X_train, y_train)
    deval = lgb.Dataset(X_val, y_val)

    params = {'objective':'binary',
              'learning_rate':0.05,
              'metric':['auc','binary_logloss'],
              'max_depth':7,
              'num_leaves':70,
              'verbose':-1
             }
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain, deval], callbacks=[early_stopping(50), log_evaluation(500)])

    # feature importance
    feature_importance_df = pd.DataFrame({'feature_name': model.feature_name(),'tree_split': model.feature_importance(importance_type='split')})
    feature_importance_df = feature_importance_df.sort_values(by='tree_split', ascending=False)
    feature_importance_df['lgb_rank'] = feature_importance_df['tree_split'].rank(method='first',ascending=False).astype(int)

    # shap value
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    means = [np.abs(shap_values[class_]).mean(axis=0) for class_ in range(len(shap_values))] #2개의 2d array를 각각 행 방향으로 mean값 구함
    shap_means = np.sum(np.column_stack(means), 1) # 2개의 1d array를 sum해서 1개의 array로 만듦

    # 결과 dataframe 생성
    importance_df = pd.DataFrame({'feature_name': X_val.columns, 'mean_shap_value': shap_means}).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)
    importance_df['shap_rank'] = importance_df['mean_shap_value'].rank(method='first',ascending=False).astype(int)

    feature_importance_df = feature_importance_df.merge(importance_df,on='feature_name')
    return feature_importance_df

