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


#%%
# i = np.random.randint(0, len(power_df.columns))
i = 3002
data_raw = power_df.iloc[:,i]
plt.plot(data_raw)
plt.xticks(rotation=30)
plt.title(f'{power_df.columns[i]}')
plt.show()

plt.figure()
plt.plot(data_raw.values.reshape(-1,48).T,'k',alpha=0.1)
plt.plot(np.nanmedian(data_raw.values.reshape(-1,48),axis=0),'r-')
plt.xticks(range(48)[::4], range(24)[::2])
plt.legend()
plt.xlabel('Time [hour]')
plt.ylabel('Energy consumption [kWh]')
plt.title(f'{power_df.columns[i]}')
# plt.ylim(0, 2)
plt.show()


#%%
plt.figure()
i = np.random.randint(0, len(power_df.columns))
data_raw = power_df.iloc[:,i]
plt.title(f'{power_df.columns[i]}')
count_w, count_nw = 0, 0
for j in range(7):
    if weekday_flag[j] <5:
        if count_w == 0:
            plt.plot(data_raw.values.reshape(-1,48)[j,:], color='r',alpha=0.4, label='weekday')
            count_w += 1
        else:
            plt.plot(data_raw.values.reshape(-1, 48)[j, :], color='r', alpha=0.4)
    else:
        if count_nw == 0:
            plt.plot(data_raw.values.reshape(-1, 48)[j, :], color='k', alpha=0.4, label='weekend')
            count_nw += 1
        else:
            plt.plot(data_raw.values.reshape(-1, 48)[j, :], color='k', alpha=0.4)
plt.legend(loc='upper left')
plt.xticks(range(48)[::4], range(24)[::2])
plt.xlabel('Time [hour]')
plt.ylabel('Energy consumption [kWh]')
plt.ylim(0, 3)
plt.show()

#%% 집 하나에 대해서
power_dict = dict(keys=power_df.columns)

# NaN이 하나라도 있으면 drop
power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)

a = []
for i, key in enumerate(power_dict.keys()):
    power_dict[key] = power_rs[i,non_nan_idx[i],:]
    a.append(np.median(np.mean(power_rs[i, non_nan_idx[i], :], axis=0)))


#%%
plt.title('Number of valid days')
plt.xlabel('Index of house')
plt.ylabel('days')
plt.plot(non_nan_idx.sum(axis=1))
plt.show()
