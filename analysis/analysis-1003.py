# %%
'''
1: number of residents
2: number of appliances
3: single or not
4: retired or not
'''
option = int(input('Option number?'))

if option == 1:
    mrmr_features_cer = np.array([1, 32])
    mrmr_features_save = np.array([3, 9, 4])
elif option == 2:
    # NOT IMPLEMENTED YET
    pass
elif option == 3:
    mrmr_features_cer = [38, 24, 36, 31]
    mrmr_features_save = [4, 8]
elif option == 4:
    mrmr_features_cer = [39,41,38, 40]
    mrmr_features_save = [9, 7, 8, 10]

def quantiz(label, option):
    if option == 1:
        label[label<=2] = 0
        label[label>2] = 1
    elif option == 2:
        label[label<=8] = 0
        label[label>11] = 2
        label[label>8] = 1
    elif option == 3:
        label[label==1] = 0
        label[label!=0] = 1
    elif option == 4:
        pass
    return label


# %%
import os
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
CER_label = pd.read_csv('data/CER/survey_processed_0728.csv')
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
        if option == 1:
            label_raw = CER_label['Q13'].values # number of residents
        elif option == 2:
            label_raw = CER_label['Q12'].values # number of appliances
        elif option == 3:
            label_raw = CER_label['Q13'].values # number of residents
        elif option == 4:
            label_raw = CER_label['Q2'].values # employed or not 
            label_raw = (label_raw == 0).astype(int)

            label_2 = CER_label['Q13'].values
        home_arr = home_arr_c
    elif name == 'SAVE':
        data_raw = SAVE_rs
        if option == 1:
            label_raw = SAVE_label['Q2'].values # number of residents
        elif option == 2:
            label_raw = SAVE_label.loc[:,'Q3_19_1':'Q3_19_33'].sum(axis=1).values # number of appliances
        elif option == 3:
            label_raw = SAVE_label['Q2'].values # number of residents
        elif option == 4:
            label_raw = SAVE_label['Q2D'].values # retired or not
            label_raw = (label_raw == 7).astype(int)

            label_2 = SAVE_label['Q2'].values


        home_arr = home_arr_s

    invalid_idx = pd.isnull(label_raw)
    invalid_idx += pd.isnull(label_2)
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

    label_2 = np.array([label_2[u] for u in unique_home_arr])

    data_dict[name] = [data_raw, label, home_arr, label_2]

# %% Daily typical load generation

rep_load_dict = dict()
label_dict = dict()
for name in ['SAVE','CER']:
    data_raw, label, home_arr, label_2 = data_dict[name]
    unique_home_arr = np.unique(home_arr)
    rep_load_list = []
    for i, home_idx in enumerate(unique_home_arr):
        data = data_raw[home_arr == home_idx,:]
        rep_load_list.append(data.mean(axis=0))
    rep_load_dict[name] = np.array(rep_load_list)
    label_dict[name] = (label, label_2)

# %%
n_res = 1
for data_name in ['SAVE', 'CER']:
    plt.title(data_name)
    idx = np.all([label_dict[data_name][0] == 1, label_dict[data_name][1] == n_res], axis=0)
    data_1 = rep_load_dict[data_name][idx].reshape(-1)
    idx = np.all([label_dict[data_name][0] == 0, label_dict[data_name][1] == n_res], axis=0)
    data_2 = rep_load_dict[data_name][idx].reshape(-1)
    plt.hist(data_1, label = 'retired', alpha = 0.5, density = True, bins = 100)
    plt.hist(data_2, label = 'not retired', alpha = 0.5, density = True, bins = 100)
    plt.legend()
    plt.xlim(0, 3)
    plt.show()