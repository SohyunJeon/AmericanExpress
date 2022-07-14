"""
- feature 경로 : G:\공유 드라이브\BRIQUE\(B)과제\(BB)내부\기타\BI-Y20-DS-005-1 TraceDataAnalysis\03.분석\feature_store
- feature 명: paydate_adv_1.csv
- 용량 : 1 GB
- 설명 : S_2(날짜) 변수에 대한 다양한 파생변수
    days_btw_pay_{통계량} : 개별 고객의 지불 일자의 차이에 대한 통계값
    days_btw_pays_last2, 3 : 개별 고객의 지불 일자의 차이에 대해 마지막으로 부터 2,3번째 앞의 값
    S_2_last : S_2 마지막 값
    S_2_first_diff:  개별 고객의 지불 일자의 차이에 대한 처음값
    S_2_last_diff:  개별 고객의 지불 일자의 차이에 대한 마지막값
    S_2_{first/last}_{dd/mm/yy} : S_2 fist, last값을 datetime 분해

"""


#%%
import pandas as pd
from pyarrow import feather


def last2(series):
    return series.values[-2] if len(series.values) >= 2 else -127


def last3(series):
    return series.values[-3] if len(series.values) >= 3 else -127


def get_paydate_features(data: pd.DataFrame) -> pd.DataFrame:
    groupby_col_name = "customer_ID"
    df = data.loc[:, [groupby_col_name, "S_2"]]

    # 고객당 지불 일자별 기간에 대한 summary
    statements = df.groupby(groupby_col_name)["S_2"].agg("diff").dt.days
    statements = statements.fillna(0)
    statements_df = pd.DataFrame(statements.values, columns=["days_btw_pay"])
    statements_df.insert(0, groupby_col_name, df[groupby_col_name])

    days_btw_pay = statements_df.groupby(groupby_col_name).agg(
        ["mean", "std", "max", "min"]
    )
    days_btw_pay.columns = ["_".join(x) for x in days_btw_pay.columns]

    # 고객별 지불 일자의 max(고객 천체 처음값) - 처음값 /  min(고객 전체 마지막값) - 마지막값
    days_fl = df.groupby(groupby_col_name).agg(["first", "last"])
    days_fl.columns = ["_".join(x) for x in days_fl.columns]

    days_fl["S_2_first_diff"] = (
        days_fl["S_2_first"].min() - days_fl["S_2_first"]
    ).dt.days
    days_fl["S_2_last_diff"] = (days_fl["S_2_last"].max() - days_fl["S_2_last"]).dt.days

    # 마지막 일자의 datetime 분해
    for d_col in ["first", "last"]:
        days_fl[f"S_2_{d_col}_dd"] = days_fl[f"S_2_{d_col}"].dt.day
        days_fl[f"S_2_{d_col}_mm"] = days_fl[f"S_2_{d_col}"].dt.month
        days_fl[f"S_2_{d_col}_yy"] = days_fl[f"S_2_{d_col}"].dt.year

    days_fl = days_fl.drop("S_2_first", axis=1)

    # 지불 일자의 한달 전, 두달전 값
    last_more = df.groupby(groupby_col_name).agg([last2, last3])
    last_more.columns = ["_".join(x) for x in last_more.columns]

    statements_last_more = statements_df.groupby(groupby_col_name).agg([last2, last3])
    statements_last_more.columns = ["_".join(x) for x in statements_last_more.columns]

    # 취합
    total_result = pd.concat(
        [days_btw_pay, statements_last_more, days_fl, last_more], axis=1
    ).reset_index()

    return total_result


if __name__ == "__main__":
    data = feather.read_feather("./SH/data/train_data.ftr")
    result = get_paydate_features(data)
