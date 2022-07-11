"""
- feature 경로 : G:\공유 드라이브\BRIQUE\(B)과제\(BB)내부\기타\BI-Y20-DS-005-1 TraceDataAnalysis\03.분석\feature_store
- feature 명: paydate_basic.parquet
- 용량 : 27.4MB
- 설명 : 각 명세서 월을 int변환 (시작월 : 0 ~) -> 변환 후 first, last, period(max - last) 산출
"""

#%%
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


#%%

def period(x: np.array) -> np.array:
    return x.max() - x.min()


def get_month_index(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """
    datetime 컬럼에서 month 기준, 시간 순서대로 int로 변환 (첫달 -> 0)
    :param df:
    :param date_col_name: datetime 컬럼명
    :return:
    """
    start_month = datetime.strftime(df[date_col_name].min(), '%Y-%m')
    end_month = datetime.strftime(df[date_col_name].max() + relativedelta(months=1), '%Y-%m')
    time_range = pd.date_range(start_month, end_month, freq='M', inclusive='right')
    time_dict = {datetime(k.year, k.month, 1): v for k, v in zip(time_range, range(0, len(time_range)))}
    df['paymonth_idx'] = df[date_col_name].apply(lambda x: time_dict[datetime(x.year, x.month, 1)])
    df = df.drop(date_col_name, axis=1)
    return df


def period(x: np.array) -> np.array:
        return x.max() - x.min()


def get_payment_basic_feats(df: pd.DataFrame, date_col_name: str, groupby_col_name: str) -> pd.DataFrame:
    """
    :param df: groupby 진행할 dataframe
    :param date_col_name: datetime 컬럼 명
    :param groupby_col_name: groupby 기준 컬럼 명
    :return: groupby 결과
    """
    df = df.loc[:, [groupby_col_name, date_col_name]]
    # change payment month to int
    df = get_month_index(df, date_col_name)

    # agg: first, last, period
    result = df.groupby(groupby_col_name).agg(['first', 'last', period])
    result.columns = ['_'.join(x) for x in result.columns]
    result = result.reset_index()

    return result



if __name__ == '__main__':
    data = pd.read_feather('./SH/data/train_data.ftr')
    train_payment_feats = get_payment_basic_feats(data, date_col_name='S_2', groupby_col_name='customer_ID')