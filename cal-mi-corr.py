import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import *
font = {'size': 14}
matplotlib.rc('font', **font)

#%% load dataset
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
info = load_info(path='data/survey_for_number_of_residents.csv')
datetime = power_df['time']
datetime = pd.to_datetime(datetime)

power_df.set_index('time', inplace=True)
info[info>=7] = 7

label = info['count'].values
label = label.astype(int)

home_n = power_df.shape[1]

#%% 대표 프로파일로 변환
type = 3 # 1: 전체 2: 주중 3: 주말
power_arr = power_df.values
# 주말만
weekday_flag = np.reshape(datetime.dt.weekday.values, (-1,48))[:,0]
if type == 1:
    weekday_flag[:] = True
elif type == 2:
    weekday_flag = [w >= 5 for w in weekday_flag]
elif type == 3:
    weekday_flag = [w < 5 for w in weekday_flag]
weekday_flag = np.array(weekday_flag).astype(bool)
power_arr_rs = np.reshape(power_arr.T, (home_n,-1,48)) # home, day, hours
power_arr_rs = power_arr_rs[:,weekday_flag,:]
power_arr_rs = np.nanmean(power_arr_rs, axis=1)
power_arr = power_arr_rs
del power_arr_rs

# 24시간 -> mean 으로 reduction
mi_list, corr_list = [], []
for i in range(48):
    data = power_arr[:,i]
    data[pd.isna(data)] = 0
    result_0, result_1 = calc_MI_corr(data, label)
    mi_list.append(result_0)
    corr_list.append(result_1)

fig, ax1 = plt.subplots(figsize=(6,3))
plt.xticks(list(range(48)[::4])
           , list(np.arange(0,24,0.5)[::4].astype(int)))
# plt.title('Correlation and MI')
# ax1.set_xlabel('date')
ax1.plot(mi_list, color='tab:blue')
ax1.set_ylabel('MI',color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue', )
ax2 = ax1.twinx()
ax2.plot(corr_list,color='tab:orange')
ax2.set_ylabel('Correlation',color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
# fig.tight_layout()
ax1.set_xlabel('hour',color='k')
plt.show()
print(corr_list)

#%% feature에 대해
mi_list, corr_list = [], []
# mean
data = np.nanmean(power_arr, axis=1)
data[pd.isna(data)] = 0
result_0, result_1 = calc_MI_corr(data, label)
mi_list.append(result_0)
corr_list.append(result_1)

# median
data = np.nanmedian(power_arr, axis=1)
data[pd.isna(data)] = 0
result_0, result_1 = calc_MI_corr(data, label)
mi_list.append(result_0)
corr_list.append(result_1)

# max
data = np.nanmax(power_arr, axis=1)
data[pd.isna(data)] = 0
result_0, result_1 = calc_MI_corr(data, label)
mi_list.append(result_0)
corr_list.append(result_1)

fig, ax1 = plt.subplots(figsize=(6,3))
mi_cord = np.array([0, 3, 6])
corr_cord = np.array([1, 4, 7])
plt.xticks((mi_cord + corr_cord) / 2, ['daily mean', 'daily median', 'daily max'])
ax1.bar(mi_cord, mi_list, label='MI',color='tab:blue')
ax1.set_ylabel('MI',color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue', )
ax2 = ax1.twinx()
ax2.bar(corr_cord, corr_list, label='Corr',color='tab:orange')
ax2.set_ylabel('Correlation',color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
# plt.legend()
plt.show()