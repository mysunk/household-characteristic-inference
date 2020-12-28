import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import *
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

# 데이터 로드 및 전처리
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
power_df['time'] = pd.to_datetime(power_df['time'])
weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)

#%% NaN이 하나라도 있으면 drop
power_dict = dict()

power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)

a = []
for i, key in enumerate(power_df.columns):
    # 최소 5일 이상 valid day가 있을 때 추가
    if non_nan_idx.sum() > 5:
        power_dict[key] = power_rs[i,non_nan_idx[i],:]

#%%
plt.title('Number of valid days')
plt.xlabel('Index of house')
plt.ylabel('days')
plt.plot(non_nan_idx.sum(axis=1))
plt.show()