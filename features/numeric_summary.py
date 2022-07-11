"""
feature 경로 : G:\공유 드라이브\BRIQUE\(B)과제\(BB)내부\기타\BI-Y20-DS-005-1 TraceDataAnalysis\03.분석\feature_store
feature 명: numeric_basic_summary.parquet
용량 : 1.54GB
설명 : 연속형 변수에 대한 다음 항목의 summary 변수
    평균, 표준편차, 최소, 최대, 처음 값, 마지막 값
"""

#%%
import pandas as pd



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
