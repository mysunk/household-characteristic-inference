# %%

import os
os.chdir('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
font = {'size': 16, 'family':"Malgun Gothic"}
matplotlib.rc('font', **font)

from pathlib import Path
from module.util_0607 import *
import pandas as pd

# deep learning 관련
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn import metrics
import numpy as np
import scipy.io
import scipy.linalg
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

### 에너지 데이터 로드
SAVE = pd.read_csv('data/SAVE/power_0428.csv', index_col=0)
SAVE = SAVE.iloc[84:,:]
SAVE.index = pd.to_datetime(SAVE.index)
SAVE[SAVE == 0] = np.nan
SAVE = SAVE.loc[pd.to_datetime('2017-01-01 00:00'):,:]

helper_dict = defaultdict(list)
for col in SAVE.columns:
    helper_dict[col[2:]].append(col)

# 동일 집끼리 병합
drop_cols = []
invalid_idx_list = []
for key,value in helper_dict.items():
    if len(value) >= 2:
        valid_idx_1 = ~pd.isnull(SAVE[value[1]])

        # replace value
        SAVE[value[0]][valid_idx_1] = SAVE[value[1]][valid_idx_1]

        # delete remain
        drop_cols.append(value[1])

# drop cols
SAVE.drop(columns = drop_cols, inplace = True)

# label과 data의 column 맞춤
SAVE.columns = [c[2:] for c in SAVE.columns]

### 라벨 로드
SAVE_label = pd.read_csv('data/SAVE/save_household_survey_data_v0-3.csv', index_col = 0)
# Interviedate가 빠른 순으로 정렬
SAVE_label.sort_values('InterviewDate', inplace = True)
SAVE_label = SAVE_label.T
SAVE_label.columns = SAVE_label.columns.astype(str)

# 라벨 순서를 데이터 순서와 맞춤
valid_col = []
for col in SAVE.columns:
    if col in SAVE_label.columns:
        valid_col.append(col)

SAVE_label = SAVE_label[valid_col].T
SAVE = SAVE[valid_col]
print('Done load SAVE')
SAVE[SAVE == 0] = np.nan

# CER 데이터 로드
# start_date = pd.to_datetime('2010-09-01 00:00:00')
# end_date = pd.to_datetime('2009-12-01 23:00:00')

power_df = pd.read_csv('data/CER/power_comb_SME_included.csv')

# 0 to NaN
power_df[power_df==0] = np.nan
power_df['time'] = pd.to_datetime(power_df['time'])
power_df.set_index('time', inplace=True)

# load label
CER_label = pd.read_csv('data/CER/survey_processed_0427.csv')
CER_label['ID'] = CER_label['ID'].astype(str)
CER_label.set_index('ID', inplace=True)

CER = power_df.loc[:,CER_label.index]
del power_df
print('Done load CER')

## 기간을 6개월 한정
start_date = pd.to_datetime('2018-01-01 00:00:00')
end_date = pd.to_datetime('2018-06-30 23:45:00')

SAVE = SAVE.loc[start_date:end_date,:]

start_date = pd.to_datetime('2010-01-01 00:00:00')
end_date = pd.to_datetime('2010-06-30 23:30:00')

CER = CER.loc[start_date:end_date,:]

# Downsampling SAVE
n = SAVE.shape[0]
list_ = []
time_ = []
for i in range(0, n, 2):
    data = SAVE.iloc[i:i+2,:]
    invalid_data_idx = np.any(pd.isnull(data), axis=0)
    data = data.sum(axis=0)
    data.iloc[invalid_data_idx] = np.nan
    list_.append(data)
    time_.append(SAVE.index[i])
list_ = pd.concat(list_, axis=1).T
list_.index = time_
SAVE = list_
del list_

nan_ratio = pd.isnull(SAVE).sum(axis=0) / SAVE.shape[0]
invalid_idx = (nan_ratio == 1)
SAVE = SAVE.loc[:,~invalid_idx]
SAVE_label = SAVE_label.loc[~invalid_idx,:]

# invalid house processing
nan_ratio = pd.isnull(CER).sum(axis=0) / CER.shape[0]
invalid_idx = (nan_ratio == 1)
CER = CER.loc[:,~invalid_idx]
CER_label = CER_label.loc[~invalid_idx,:]

print(CER.shape)
print(CER_label.shape)

print(SAVE.shape)
print(CER.shape)

# %% 
# 2d daily 형태로 변환 (house * day , hour)
CER_rs, home_arr_c = transform(CER, 24 * 2)
SAVE_rs, home_arr_s = transform(SAVE, 24 * 2)

data_dict = dict()

for name in ['CER','SAVE']:
    if name == 'CER':
        data_raw = CER_rs
        # label_raw = CER_label['Q13'].values
        label_raw = CER_label['Q12'].values
        home_arr = home_arr_c
    elif name == 'SAVE':
        data_raw = SAVE_rs
        label_raw = SAVE_label['Q2'].values
        home_arr = home_arr_s

    invalid_idx = pd.isnull(label_raw)
    # label_raw = label_raw[~invalid_idx]
    invalid_home_num = np.where(invalid_idx)[0]
    valid_home_idx = np.zeros(home_arr.shape, dtype = bool)
    for j in range(home_arr.shape[0]):
        if home_arr[j] in invalid_home_num:
            valid_home_idx[j] = False
        else:
            valid_home_idx[j] = True
    data_raw = data_raw[valid_home_idx,:]
    home_arr = home_arr[valid_home_idx]
    unique_home_arr = np.unique(home_arr)
    label = np.array([label_raw[u] for u in unique_home_arr])

    data_dict[name] = [data_raw, label, home_arr]

rep_load_dict = dict()
label_dict = dict()
for name in ['CER','SAVE']:
    data_raw, label, home_arr = data_dict[name]
    unique_home_arr = np.unique(home_arr)
    rep_load_list = []
    for i, home_idx in enumerate(unique_home_arr):
        data = data_raw[home_arr == home_idx,:]
        rep_load_list.append(data.mean(axis=0))
    rep_load_dict[name] = np.array(rep_load_list)
    label_dict[name] = label

# %% Daily typical load generation (CER 만)
import multiprocessing
from functools import partial
import time

data_raw, label, home_arr = data_dict['CER']

def helper(i):
    idx = home_arr == i
    data = data_raw[idx,:]
    n_samples = data.shape[0]
    distance = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance[i, j] = np.linalg.norm(data[i,:]-data[j,:])

    # rep_load_idx = np.argmin(distance.sum(axis=0))
    return distance

PROCESSES = multiprocessing.cpu_count()
print('{}개의 CPU 사용...'.format(PROCESSES))

start = time.time()
if __name__ == '__main__':
    
    unique_home_arr = np.unique(home_arr)
    p = multiprocessing.Pool(processes = PROCESSES)
    distance_list = p.map(helper, unique_home_arr)

end = time.time()
print('Elapsed {:.2f} hours..'.format((end - start) / 3600))
# %%
rep_load_list = []
for i, home_idx in enumerate(unique_home_arr):
    distance = distance_list[i].mean(axis=0)
    mean_ = distance.mean()
    std_ = np.sqrt(distance.var())
    data = data_raw[home_arr == home_idx,:]

    idx = distance <= mean_ - 0.1 * std_

    data = data[idx,:]
    rep_load_list.append(data.mean(axis=0))
    break

# rep_load_dict['CER_p'] = np.array(rep_load_list)

# %% 시각화 비교
plt.figure(figsize = (10, 3))
plt.plot(CER.iloc[:48*7,0].values)

for i in range(7):
    if idx[i] == True:
        pass
    else:
        color = 'r'
        plt.axvspan(i * 48, (i+1) * 48, facecolor=color, alpha=0.3, label = 'Outlier')
    
plt.xticks(np.arange(24, 48 * 7  + 24, 48), [f'Day {i}' for i in range(1, 8)])
# plt.xlabel('Date')
plt.ylabel('Energy [kW]')
plt.legend()
plt.show()

# %% Instance selection
def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence

#  CER 하나씩 제거해가며 SAVE와 KL divergence 비교 
from scipy.stats import norm

kl_result = []
n_data = rep_load_dict['CER'].shape[0]
for i in range(n_data):
    a = list(range(n_data))
    a.remove(i)
    cer_data  = rep_load_dict['CER'][a,:]
    save_data = rep_load_dict['SAVE']

    mean_1, std_1 = np.mean(cer_data), np.std(cer_data)
    mean_2, std_2 = np.mean(save_data), np.std(save_data)

    x = np.arange(-10, 10, 0.001)    
    p = norm.pdf(x, mean_1, std_1)
    q = norm.pdf(x, mean_2, std_2)

    kl_val = KL(p, q)
    kl_result.append(kl_val)

# reference
cer_data  = rep_load_dict['CER']
save_data = rep_load_dict['SAVE']

mean_1, std_1 = np.mean(cer_data), np.std(cer_data)
mean_2, std_2 = np.mean(save_data), np.std(save_data)

x = np.arange(-10, 10, 0.001)    
p = norm.pdf(x, mean_1, std_1)
q = norm.pdf(x, mean_2, std_2)

ref= KL(p, q)

kl_result_2 = []
n_data = rep_load_dict['SAVE'].shape[0]
for i in range(n_data):
    a = list(range(n_data))
    a.remove(i)
    cer_data  = rep_load_dict['CER']
    save_data = rep_load_dict['SAVE'][a,:]

    mean_1, std_1 = np.mean(cer_data), np.std(cer_data)
    mean_2, std_2 = np.mean(save_data), np.std(save_data)

    x = np.arange(-10, 10, 0.001)    
    p = norm.pdf(x, mean_1, std_1)
    q = norm.pdf(x, mean_2, std_2)

    kl_val = KL(p, q)
    kl_result_2.append(kl_val)

# reference
cer_data  = rep_load_dict['CER']
save_data = rep_load_dict['SAVE']

mean_1, std_1 = np.mean(cer_data), np.std(cer_data)
mean_2, std_2 = np.mean(save_data), np.std(save_data)

x = np.arange(-10, 10, 0.001)    
p = norm.pdf(x, mean_1, std_1)
q = norm.pdf(x, mean_2, std_2)

ref_2= KL(p, q)
filtered_idx_1 = np.array(kl_result) > ref
filtered_idx_2 = np.array(kl_result_2) > ref_2

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(rep_load_dict['CER'])
data_1 = pca.transform(rep_load_dict['CER'])
data_2 = pca.transform(rep_load_dict['SAVE'])

# %%
plt.plot(data_1[filtered_idx_1,0],data_1[filtered_idx_1,1],'.', label = 'Selected source')
# plt.plot(data_2[:,0],data_2[:,1],'.', label = 'Target')
plt.plot(data_1[~filtered_idx_1,0],data_1[~filtered_idx_1,1],'rx', label = 'Removed source',alpha = 0.3, 
markersize = 3)
plt.xlabel('PC 1 (67%)')
plt.ylabel('PC 2 (0.07%)')
plt.legend()
plt.show()

