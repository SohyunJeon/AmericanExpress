"""
- feature 경로 : G:\공유 드라이브\BRIQUE\(B)과제\(BB)내부\기타\BI-Y20-DS-005-1 TraceDataAnalysis\03.분석\feature_store
- feature 명: paydate_gap.parquet
- 용량 : 27MB
- 설명 : 각 명세서 월을 int변환 (시작월 : 0 ~)
        -> 변환 후 first, last, period(max - min), count 산출
        -> 고객이 명세서를 발행받기 시작한 첫 달부터 마지막 달까지 빠지는 달이 있는지를 gap으로 산출. 있으면 1, 없으면 0.
"""

#%%
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import fastparquet

def period(x: np.array) -> np.array:
    return x.max() - x.min()


def get_month_index(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """
    datetime 컬럼에서 month 기준, 시간 순서대로 int로 변환 (첫달 -> 0)
    :param df:
    :param date_col_name: datetime 컬럼명
    :return:
    """
    start_month = datetime.strftime(df[date_col_name].min(), "%Y-%m")
    end_month = datetime.strftime(
        df[date_col_name].max() + relativedelta(months=1), "%Y-%m"
    )
    time_range = pd.date_range(start_month, end_month, freq="M", inclusive="right")
    time_dict = {
        datetime(k.year, k.month, 1): v
        for k, v in zip(time_range, range(0, len(time_range)))
    }
    df["paymonth_idx"] = df[date_col_name].apply(
        lambda x: time_dict[datetime(x.year, x.month, 1)]
    )
    df = df.drop(date_col_name, axis=1)
    return df



def get_payment_basic_feats(
    df: pd.DataFrame, date_col_name: str, groupby_col_name: str
) -> pd.DataFrame:
    """
    :param df: groupby 진행할 dataframe
    :param date_col_name: datetime 컬럼 명
    :param groupby_col_name: groupby 기준 컬럼 명
    :return: groupby 결과
    """
    df = df.loc[:, [groupby_col_name, date_col_name]]
    # change payment month to int
    df = get_month_index(df, date_col_name)

    # agg: first, last, period, count
    result = df.groupby(groupby_col_name).agg(["first", "last", period, "count"])
    result.columns = ["_".join(x) for x in result.columns]
    result = result.reset_index()

    return result


def get_payment_basic_and_gap_feats(df: pd.DataFrame, date_col_name: str, groupby_col_name: str) -> pd.DataFrame:
    """
    :param df: groupby 진행할 dataframe
    :param date_col_name: datetime 컬럼 명
    :param groupby_col_name: groupby 기준 컬럼 명
    :return: customer_ID,paymonth_gap 컬럼을 가진 dataframe
    """
    # agg dataframe 만들기
    payment_basic_df = get_payment_basic_feats(df,date_col_name,groupby_col_name)

    result_df = payment_basic_df.copy()
    result_df['paymonth_idx_gap'] = float('NaN')

    # 모든 고객이 매 달 명세서가 있지 않음. 실제 명세서 첫 달과 끝 달 사이에 없는 명세서 달이 있으면 gap 이 있다고 간주
    # period 는 'max' - 'min' 를 의미하므로 period는 끝 달 - 첫 달 사이에 몇 개월이 흘렀는지를 나타냄.
    # period == count-1 이면 gap = 0 , period != count-1 이면 gap = 1
    condition_gap = payment_basic_df['paymonth_idx_period'] > payment_basic_df['paymonth_idx_count'] - 1
    condition_no_gap = payment_basic_df['paymonth_idx_period'] == payment_basic_df['paymonth_idx_count'] - 1

    result_df.loc[condition_gap, ('paymonth_idx_gap')] = 1
    result_df.loc[condition_no_gap, ('paymonth_idx_gap')] = 0
    result_df = result_df.loc[:,('customer_ID','paymonth_idx_gap')]
    return result_df


if __name__ == "__main__":
    data = pd.read_feather(r"G:\내 드라이브\code\amex_default_predict\data\train_data.ftr")
    test_data = pd.read_feather(r"G:\내 드라이브\code\amex_default_predict\data\test_data.ftr")
    test_payment_feats = get_payment_basic_and_gap_feats(test_data, date_col_name="S_2", groupby_col_name="customer_ID")
    train_payment_feats = get_payment_basic_and_gap_feats(data, date_col_name="S_2", groupby_col_name="customer_ID")
    # filepath = r'G:\내 드라이브\code\amex_default_predict\features\paydate_gap.parquet'
    # train_payment_feats.to_parquet(filepath, engine='fastparquet')