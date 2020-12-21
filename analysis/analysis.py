import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import *
font = {'size': 14}
matplotlib.rc('font', **font)

#%% load dataset
power_df = pd.read_csv('../data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
# power_df = power_df.iloc[:2*24*28,:] # 4주
info = load_info(path='../data/survey_for_number_of_residents.csv')
datetime = power_df['time']
power_df.set_index('time', inplace=True)
# info[info>=7] = 7

power_df = power_df.values
label = info['count'].values
label = label.astype(int)

drop_idx = [180, 1049, 1101, 3535, 3927]
rest_idx = list(range(label.shape[0]))
for idx in reversed(drop_idx):
    del rest_idx[idx]
rest_idx = np.array(rest_idx)

power_df = power_df[:,rest_idx]
label = label[rest_idx]

#%% boxplot
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

data = np.nanmean(power_df, axis=0)
# NaN to 0
data[pd.isna(data)] = 0
data = data.astype(float)

data_list = []
for v in np.sort(np.unique(label)):
    data_list.append(data[label==v])

plt.boxplot(data_list)
plt.xlabel('이용객 수')
plt.ylabel('평균 전력 사용량 [kWh]')
plt.show()

print(f'Correlation is {np.corrcoef(np.ravel(data), np.ravel(label))}')

#%% 시간별
# list containing result
MI_result = []
corr_result = []

calc_MI_corr(list(range(0,11)) + list(range(21,48)))
calc_MI_corr(range(48))

MI_result = np.array(MI_result)
corr_result = np.array(corr_result)

#%%
result_df = pd.read_csv('../results/result_1.csv')
result_df.set_index('Unnamed: 0', inplace = True)

MI_result = result_df.loc['MI',:].values
corr_result = result_df.loc['Corr',:].values

#%% 계절별 corr, MI
label = info['count'].values
label = label.astype(int)

days = int(power_df.shape[0] / 48)
window = 2*24 # 하루

corr_result = np.zeros((365*2))
MI_result = np.zeros((365*2))

# 하루 단위로..
for i, start in enumerate(range(0, window*(days-1),window)):
    print(f'{i}...')
    MI_, corr_ = calc_MI_corr(power_df, label, range(start, start + window, 1))
    corr_result[i] = corr_
    MI_result[i] = MI_

#%% plot corr and mi
result_df = pd.read_csv('../results/result_1.csv')
result_df.set_index('Unnamed: 0', inplace = True)

MI_result = result_df.loc['MI',:].values
corr_result = result_df.loc['Corr',:].values

fig, ax1 = plt.subplots(figsize=(6,3))
plt.xticks(range(48)[::4], np.arange(0,24,0.5)[::4].astype(int))
plt.title('Correlation and MI')
# ax1.set_xlabel('date')
ax1.plot(MI_result, color='k')
ax1.set_ylabel('MI',color='k')
ax1.tick_params(axis='y', labelcolor='k', )
ax2 = ax1.twinx()
ax2.plot(corr_result,color='m')
ax2.set_ylabel('Correlation',color='m')
ax2.tick_params(axis='y', labelcolor='m')
# fig.tight_layout()
ax1.set_xlabel('hours',color='k')
plt.show()

#%% NaN analysis
plt.plot(pd.isna(power_df).sum(axis=1))
datetime = pd.to_datetime(datetime)
datetime_tmp = datetime.dt.strftime('%Y-%m-%d')
plt.xticks(range(len(datetime_tmp))[::3000], datetime_tmp[::3000], rotation=30)
plt.ylabel('Number of NaN')
plt.title('')
plt.xlabel('Date')
plt.show()

#%%
calc_MI_corr(power_df, label, time_idx=range(48), reduce_way='mean') # 하루
calc_MI_corr(power_df, label, time_idx=range(48 * 7), reduce_way='mean') # 일주일
calc_MI_corr(power_df, label, time_idx=range(48* 7 * 4), reduce_way='mean') # 한달

#%% histogram
values = np.unique(label)
for v in values:
    col = label == v
    data = power_df.iloc[:,col].copy()
    data = data[~pd.isnull(data)]
    plt.hist(np.ravel(data), label=v, bins=50, density=True, histtype='step')
plt.ylabel('Density')
plt.xlabel('Energy [kWh]')
plt.legend()
plt.show()


#%%
if reduce_way == 'median':
    data = np.nanmedian(power_df.copy()[time_idx, :], axis=0)
elif reduce_way == 'mean':
    data = np.nanmean(power_df.copy()[time_idx, :], axis=0)
elif reduce_way == 'max':
    data = np.nanmax(power_df.copy()[time_idx, :], axis=0)

data[pd.isna(data)] = 0

#%% proportion 최적화 1
power_df_sub = power_df[:,:48*7].copy()
search_space = np.arange(0.1, 2, 0.1)
results = np.zeros((search_space.shape[0], 2))
for i, lower_th in enumerate(search_space):
    num_eff_points = (~pd.isnull(power_df_sub)).sum(axis=0)
    data = (power_df_sub < lower_th).sum(axis=0) / num_eff_points
    results[i, :] = calc_MI_corr(data, label)

#%% proportion 최적화 2
power_df_sub = power_df.copy()
search_space = np.arange(0.1, 2, 0.1)
results = np.zeros((search_space.shape[0], 2))
for i, upper_th in enumerate(search_space):
    num_eff_points = (~pd.isnull(power_df_sub)).sum(axis=0)
    data = (power_df_sub > upper_th ).sum(axis=0) / num_eff_points
    results[i, :] = calc_MI_corr(data, label)

#%%
fig, ax1 = plt.subplots(figsize=(6,3))
plt.xlabel('Lower bound')
ax1.plot(search_space, results[:,0], label='MI',color='tab:blue')
ax1.set_ylabel('MI',color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue', )
ax2 = ax1.twinx()
ax2.plot(search_space, results[:,1], label='Corr',color='tab:orange')
ax2.set_ylabel('Correlation',color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
# plt.legend()
plt.show()

#%% 대표 프로파일로 변환
type = 1 # 1: 전체 2: 주말 3: 주중
power_arr =power_df.copy()
datetime = pd.to_datetime(datetime)
weekday_flag = np.reshape(datetime.dt.weekday.values, (-1,48))[:,0]
if type == 1:
    weekday_flag[:] = True
elif type == 2:
    weekday_flag = [w >= 5 for w in weekday_flag]
elif type == 3:
    weekday_flag = [w < 5 for w in weekday_flag]
weekday_flag = np.array(weekday_flag).astype(bool)
power_arr_rs = np.reshape(power_arr.T, (power_df.shape[1],-1,48)) # home, day, hours
# weekday_flag = weekday_flag[:7]
power_arr_rs = power_arr_rs[:,weekday_flag,:]
power_arr = np.reshape(power_arr_rs, (power_df.shape[1], -1))
power_arr = power_arr.T

search_space = np.arange(0.1, 2, 0.1)
results = np.zeros((search_space.shape[0], 2))
for i, upper_th in enumerate(search_space):
    upper_th = 0.6
    num_eff_points = (~pd.isnull(power_arr)).sum(axis=0)
    data = (power_arr > upper_th ).sum(axis=0) / num_eff_points
    results[i, :] = calc_MI_corr(data, label)

print(results[np.argmax(results[:,1]),:])

#%%