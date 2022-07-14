import pandas as pd

# 예제코드 선정 feature: ['B_11', 'B_14', 'B_17', 'D_39', 'D_131', 'S_16', 'S_23']
colnames = [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]

def get_after_pay(df, colnames):
    for bcol in colnames:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
    return df


