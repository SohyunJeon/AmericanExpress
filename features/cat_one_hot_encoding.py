"""
- feature 경로 : G:\공유 드라이브\BRIQUE\(B)과제\(BB)내부\기타\BI-Y20-DS-005-1 TraceDataAnalysis\03.분석\feature_store
- feature 명: cat_last_ohe.parquet
- 용량 : 36MB
- 설명 : 범주형 변수의 각 ID별 마지막 값 -> One-hot-encoding 변환

"""


#%%

import pandas as pd
import numpy as np
from pyarrow import feather
from sklearn.preprocessing import OneHotEncoder


def get_cat_last_ohe(
    data: pd.DataFrame, groupby_col_name: str = "customer_ID"
) -> pd.DataFrame:
    """

    :param data:
    :param groupby_col_name: groupby 기줕 컬럼명
    :return: one-hot-encoding 결과
    """
    cat_cols = [
        "B_30",
        "B_31",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    cat_df = data.loc[:, [groupby_col_name] + cat_cols]
    cat_agg = cat_df.groupby(groupby_col_name).last()
    cat_agg.columns = ["_".join([x, "last"]) for x in cat_cols]
    cat_last = cat_agg.astype(object)

    ohe = OneHotEncoder(
        drop="first", sparse=False, dtype=np.float32, handle_unknown="ignore"
    )
    ohe.fit(cat_last)
    cat_ohe = ohe.transform(cat_last)

    result = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names_out())
    result.insert(0, groupby_col_name, cat_agg.index.values)

    return result


if __name__ == "__main__":
    data = feather.read_feather("./SH/data/train_data.ftr")
    result = get_cat_last_ohe(data)
