# %% 데이터 로드 및 전처리

import os
os.chdir('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from module.util_main import downsampling, dim_reduct
from collections import defaultdict
font = {'size': 30}
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


# %% declare functions
def DNN_model():
    x_input = Input(shape=(48,))
    x = Dense(64, activation='relu', input_shape=(48,))(x_input)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(64, activation = 'relu')(x)

    # Add svm layer
    x_out = Dense(48)(x)

    model = Model(x_input, x_out)
    optimizer = Adam(0.1, epsilon=0.1)
    # model.compile(optimizer=optimizer, loss='squared_hinge')
    model.compile(optimizer=optimizer, loss='mae', metrics =['mae', 'mse'])
    return model

# %% IDEATION 1 확인 :: 전력 예측 정확도가 transfer learning 적용 후에 향상되는지?
from tqdm import tqdm
es = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )
model_dict = dict()
for i in tqdm(range(500)): # 일단 1000개만
    data = CER.iloc[:,i].values
    data = data.reshape(-1, 48)
    X = data[:-1]
    y = data[1:]
    valid_idx = np.all(~pd.isnull(X), axis=1) * np.all(~pd.isnull(y), axis=1)
    X, y = X[valid_idx], y[valid_idx]
    n_day = X.shape[0]
    split = int(n_day * 0.75)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    model = DNN_model()
    history = model.fit(X_train, y_train, epochs=10000, \
                verbose=0, validation_data=(X_val, y_val),batch_size=128, callbacks = [es])
    model_dict['CER_'+str(i)] = model

# %%
# transfer learning test
mae_results = np.zeros((1000, 101))
for i in tqdm(range(1000)):
    data = SAVE.iloc[:,i].values
    data = data.reshape(-1, 48)
    X = data[:-1]
    y = data[1:]
    valid_idx = np.all(~pd.isnull(X), axis=1) * np.all(~pd.isnull(y), axis=1)
    X, y = X[valid_idx], y[valid_idx]
    n_day = X.shape[0]
    split_1 = 5
    split_2 = 10
    split_3 = 30
    if n_day < 30:
        continue
    X_train, y_train = X[:split_1], y[:split_1]
    X_val, y_val = X[split_1:split_2], y[split_1:split_2]
    X_test, y_test = X[split_2:split_3], y[split_2:split_3]
    
    for j in range(100):
        base_model = model_dict['CER_'+str(j)]

        model = Sequential()
        for layer in base_model.layers: # copy the layers
            model.add(layer)
        optimizer = Adam(0.01, epsilon=0.1)
        model.compile(optimizer=optimizer, loss='mae', metrics =['mae', 'mse'])
        model.fit(X_train, y_train, epochs=10000, \
                verbose=0, validation_data=(X_val, y_val),batch_size=128, callbacks = [es])
        
        y_pred = model.predict(X_test)
        mae_results[i, j] = mean_absolute_error(y_test, y_pred)
    
    # compare with self training
    model = DNN_model()
    model.fit(X_train, y_train, epochs=10000, \
                verbose=0, validation_data=(X_val, y_val),batch_size=128, callbacks = [es])
        
    y_pred = model.predict(X_test)
    mae_results[i, j+1] = mean_absolute_error(y_test, y_pred)

# %% test
from sklearn.metrics import mean_absolute_error
plt.plot(mae_results[3,:])
plt.show()

# %% calculate negative transfer ratio
neg_trans_ratio_result = []
for i in range(100):
    neg_trans_ratio = (mae_results[:40,i] > mae_results[:40,-1]).mean()
    neg_trans_ratio_result.append(neg_trans_ratio)

plt.plot(neg_trans_ratio_result)
plt.xlabel('CER idx')
plt.ylabel('Negative transfer ratio')
plt.show()

# %% compare with similarity
# 2d daily 형태로 변환 (house * day , hour)
CER_rs, home_arr_c = transform(CER, 24 * 2)
SAVE_rs, home_arr_s = transform(SAVE, 24 * 2)

data_dict = dict()

for name in ['CER','SAVE']:
    if name == 'CER':
        data_raw = CER_rs
        label_raw = CER_label['Q13'].values
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

# %% cosine similarity
cer_label = label_dict['CER']
cer_data = rep_load_dict['CER']

save_label = label_dict['SAVE']
save_data = rep_load_dict['SAVE']

from scipy.spatial.distance import cosine
from numpy.linalg import norm
tmp_result , tmp_result_2 = [], []
view = 0

for i in range(100):
    tmp_result.append(cosine(save_data[view,:], cer_data[i,:]))

for i in range(100):
    tmp_result_2.append(norm(save_data[view,:] - cer_data[i,:]))


# %%
fig, ax1 = plt.subplots(figsize = (10, 5))

ax1.plot(mae_result,'-', label='val_loss', color = 'tab:blue')
ax2 = ax1.twinx()
val = ax2.plot(tmp_result_2, '-', label='val_acc', color = 'tab:orange')

# %%
valid_idx = np.where(np.array(neg_trans_ratio_result) < 0.15)[0]

plt.hist(rep_load_dict['CER'][valid_idx,:].reshape(-1), label = 'CER', density = True, bins = 10, alpha = 0.3)
plt.hist(rep_load_dict['CER'].reshape(-1), label = 'CER', density = True, bins = 30, alpha = 0.3)
plt.hist(rep_load_dict['SAVE'].reshape(-1), label = 'SAVE', density = True, bins = 30, alpha = 0.3)
plt.legend()
plt.show()


# %% KL divergence between dataset
def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence


# %% CER 하나씩 제거해가며 SAVE와 KL divergence 비교 
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
plt.ylabel('KL Divergence')
plt.title('KL divergence btw CER and SAVE')
plt.plot(kl_result,'.')
plt.hlines(ref, 0, n_data, color = 'r',zorder=5)
plt.xlabel('House idx')
plt.show()


# %% SAVE 하나씩 제거해가며 CER과 비교
from scipy.stats import norm

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
plt.ylabel('KL Divergence')
plt.title('KL divergence btw CER and SAVE')
plt.plot(kl_result_2,'.')
plt.hlines(ref, 0, n_data, color = 'r',zorder=5)
plt.xlabel('House idx')
plt.show()

# %% daily variation
plt.figure(figsize = (12, 4))
plt.subplot(1,2,1)
plt.plot(CER.iloc[:48*7,0])
plt.xticks(rotation = 30)
plt.ylabel('Energy [kW]')
plt.xlabel('Date')
# plt.show()
plt.subplot(1,2,2)
plt.plot(CER.iloc[:,1].values.reshape(-1, 48).T, color = 'k', alpha = 0.3)
plt.plot(np.nanmean(CER.iloc[:,1].values.reshape(-1, 48), axis=0), color = 'r', label = 'average')
plt.ylabel('Energy [kW]')
plt.xticks(range(48)[::4], range(24)[::2])
plt.xlabel('Hour')
plt.show()

# %% 대표부하

def transform(df, sampling_interv = 24 * 2 * 7):
    '''
    [input]
    df: dataframe (timeseries, home)
    
    [output]
    data_2d: 2d array
    home_arr: home index array
    '''

    # dataframe => 3d numpy array
    n_d, n_h = df.shape
    n_w = n_d // sampling_interv
    n_d = n_w * sampling_interv
    df_rs = df.iloc[:n_d,:].values.T.reshape(n_h, -1, sampling_interv)

    # 3d numpy array => 2d numpy array
    n, m, l = df_rs.shape
    data_2d = df_rs.reshape(n*m, l)
    home_arr = np.repeat(np.arange(0, n), m)
    invalid_idx = np.any(pd.isnull(data_2d), axis=1)
    data_2d = data_2d[~invalid_idx, :]
    home_arr = home_arr[~invalid_idx]

    # constant load filtering
    invalid_idx = np.nanmin(data_2d, axis=1) == np.nanmax(data_2d, axis=1)
    data_2d = data_2d[~invalid_idx, :]
    home_arr = home_arr[~invalid_idx]

    return data_2d, home_arr

# 2d daily 형태로 변환 (house * day , hour)
CER_rs, home_arr_c = transform(CER, 24 * 2)
SAVE_rs, home_arr_s = transform(SAVE, 24 * 2)

# %%
import multiprocessing
from functools import partial
import time
unique_home_arr = np.unique(home_arr_s)
# label_raw = CER_label['Q13'].values

def helper(i):
    idx = home_arr_c == i
    data_raw = CER_rs[idx,:]

    n_samples = data_raw.shape[0]
    distance = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance[i, j] = np.linalg.norm(data_raw[i,:]-data_raw[j,:])

    # rep_load_idx = np.argmin(distance.sum(axis=0))
    return distance

PROCESSES = multiprocessing.cpu_count()
print('{}개의 CPU 사용...'.format(PROCESSES))

start = time.time()
if __name__ == '__main__':
    p = multiprocessing.Pool(processes = PROCESSES)
    distance_list = p.map(helper, unique_home_arr)

end = time.time()
print('Elapsed {:.2f} hours..'.format((end - start) / 3600))

def evaluate(X, y):
    '''
    대표 부하 평가 함수
    '''
    
    kf = KFold(n_splits=5, shuffle = True, random_state = 0)
    result_tmp = np.zeros(y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rfc = RandomForestRegressor()
        # rfc = SVC(kernel='rbf', degree=3)
        rfc.fit(X_train, y_train)
        rfc_predict = rfc.predict(X_test)

        # print(key)
        result_tmp[test_index] = rfc_predict
    result_mae = mean_absolute_error(y, result_tmp)
    return result_mae, rfc


# %%
from sklearn.metrics import mean_absolute_error
label_raw = SAVE_label['Q2'].values

rep_load_list = []
for i, home_idx in enumerate(unique_home_arr):
    distance = distance_list[i]
    idx = np.argsort(distance.sum(axis=0))
    data = SAVE_rs[home_arr_s == home_idx,:]
    rep_load_list.append(data.mean(axis=0))

rep_load_arr = np.array(rep_load_list)
label = np.array([label_raw[u] for u in unique_home_arr])
invalid_idx = pd.isnull(label)
rep_load_arr = rep_load_arr[~invalid_idx]
label = label[~invalid_idx]

result, rfc = evaluate(rep_load_arr, label)
print(result)

# %%
for th in [3]:
    print(th)
    rep_load_list = []
    for i, home_idx in enumerate(unique_home_arr):
        distance = distance_list[i].mean(axis=0)
        mean_ = distance.mean()
        std_ = np.sqrt(distance.var())
        data = SAVE_rs[home_arr_s == home_idx,:]

        idx = distance <= mean_ + th * std_

        data = data[idx,:]
        rep_load_list.append(data.mean(axis=0))
    rep_load_arr = np.array(rep_load_list)
    
    label = np.array([label_raw[u] for u in unique_home_arr])
    invalid_idx = pd.isnull(label)
    rep_load_arr = rep_load_arr[~invalid_idx]
    label = label[~invalid_idx]
    result, rfc = evaluate(rep_load_arr, label)
    print(result)

# %%
result_list = [0.8829571384520507, \
    0.8811039161270429, 0.8914276904101139, 0.8897471477027443,\
        0.8942213999383287, 0.8879401788467469, 0.908]

plt.plot([-0.1, 0, 0.5, 1, 1.5, 2, 3],result_list)
plt.show()

# %%
i = 0

distance = distance_list[i].mean(axis=0)
mean_ = distance.mean()
std_ = np.sqrt(distance.var())

distances = []
th_list = [-1]
for th in th_list:
    idx = distance <= mean_ + th * std_
    distances.append(distance[idx].mean())

# print(idx.sum())
# print(distance[idx].mean())
plt.plot(distances)
plt.xticks(range(len(th_list)), th_list)
plt.xlabel('threshod')
plt.ylabel('Mean distance')
plt.show()


# %%
feature_set = [range(48), range(0, 10), range(10, 18), range(18, 24),\
    range(24,30), range(30, 36), range(36, 42), range(42, 48)]

data_list = []
for feature in feature_set:
    data = rep_load_dict['CER'][:,feature].mean(axis=1)
    data_list.append(data)
data_list = np.array(data_list)
data_2d = data_list.T

mi_result, corr_result, _ = evaluate_features(data_2d, label_dict['CER'])
result_df = pd.DataFrame()
result_df['Corr'] = corr_result
result_df['MI'] = mi_result
import seaborn as sns

result_df_2 = pd.DataFrame()
result_df_2['data'] = np.concatenate([corr_result, mi_result])
result_df_2['label'] = ['Corr'] * len(corr_result) + ['MI'] * len(mi_result)
result_df_2['Time'] = (['T' + str(i) for i in range(0, 8)]) * 2

params = {'axes.labelsize': 16,'axes.titlesize':16, 'legend.fontsize': 16, \
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.title_fontsize':16}
matplotlib.rcParams.update(params)

plt.figure(figsize = (10, 10))
plt.subplot(2,1,1)
sns.barplot(x="Time", y="data", hue="label", data=result_df_2)
plt.ylabel('')
# plt.show()

# %%
result_df = pd.DataFrame()
for data_name in ['SAVE', 'CER']:
    data_list = []
    for feature in range(48):
        data = rep_load_dict[data_name][:,feature]
        data_list.append(data)
    data_list = np.array(data_list)
    data_2d = data_list.T

    invalid_idx = pd.isnull(label_dict[data_name])
    label = label_dict[data_name][~invalid_idx]
    data_2d = data_2d[~invalid_idx]

    mi_result, corr_result, _ = evaluate_features(data_2d, label)
    
    result_df[data_name + '_Corr'] = corr_result
    result_df[data_name + '_MI'] = mi_result

# %%
plt.figure(figsize = (6, 3))
plt.plot(result_df['CER_Corr'])
plt.plot(result_df['SAVE_Corr'])
plt.xticks(range(49)[::4], range(25)[::2])
plt.legend(['CER','SAVE'])
plt.grid(None)
plt.ylabel('Correlation')
plt.xlabel('Hour')
plt.show()
plt.figure(figsize = (6, 3))
plt.plot(result_df['CER_MI'])
plt.plot(result_df['SAVE_MI'])
plt.grid(None)
plt.xticks(range(49)[::4], range(25)[::2])
plt.legend(['CER','SAVE'], loc='lower right')

plt.ylabel('Mutual information')
plt.xlabel('Hour')
plt.show()

# %%
import seaborn as sns

result_df_2 = pd.DataFrame()
result_df_2['data'] = np.concatenate([corr_result, mi_result])
result_df_2['label'] = ['Corr'] * len(corr_result) + ['MI'] * len(mi_result)
result_df_2['Time'] = (['T' + str(i) for i in range(0, 8)]) * 2

params = {'axes.labelsize': 16,'axes.titlesize':16, 'legend.fontsize': 16, \
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.title_fontsize':16}
matplotlib.rcParams.update(params)

# plt.figure(figsize = (10, 5))
plt.subplot(2,1,2)
sns.barplot(x="Time", y="data", hue="label", data=result_df_2)
plt.ylabel('')
plt.legend([])
plt.show()