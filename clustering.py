import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import *
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

# 데이터 로드
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
# 4주
power_df = power_df.iloc[:2*24*28,:]
power_df['time'] = pd.to_datetime(power_df['time'])
weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)

# NaN이 하나라도 있으면 drop
power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)

#%% standardize
i = 0
data = power_rs[i,non_nan_idx[i],:].copy()

# standardize
# mean_ = np.mean(data, axis = 1).reshape(-1,1)
# std_ = np.std(data, axis = 1).reshape(-1,1)
# data = (data - mean_) / std_
# data = np.concatenate([data, mean_],axis=1)
# data = np.concatenate([data, std_], axis=1)

#%% kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
labels = kmeans.labels_

plt.plot(data[labels == 1,:48].T,'k', alpha=0.2)
plt.plot(data[labels == 1,:48].mean(axis=0),'k')
plt.plot(data[labels == 0,:48].mean(axis=0),'r')
plt.plot(data[labels == 0,:48].T,'r', alpha=0.2)

plt.xticks(range(48)[::4], range(24)[::2])
plt.xlabel('Time [hour]')
plt.ylabel('Energy consumption [kWh]')
# plt.ylim(0, 3)
plt.show()