from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print('=====================================================')

ts = Series([1, np.nan, np.nan, 8, 10], index = dates)
print(ts)
print(type(ts))

ts_intp_linear = ts.interpolate()           # 보간법 결측치 채워줌, 가급적 시계열에 사용
print(ts_intp_linear)
# DatetimeIndex(['2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04',
#                '2021-03-05'],
#               dtype='datetime64[ns]', freq=None)
# =====================================================
# 2021-03-01     1.0
# 2021-03-02     NaN
# 2021-03-03     NaN
# 2021-03-04     8.0
# 2021-03-05    10.0
# dtype: float64
# <class 'pandas.core.series.Series'>
# 2021-03-01     1.000000
# 2021-03-02     3.333333
# 2021-03-03     5.666667
# 2021-03-04     8.000000
# 2021-03-05    10.000000
# dtype: float64