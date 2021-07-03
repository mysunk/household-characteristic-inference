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
        label_raw = CER_label['Q13'].values
        # label_raw = CER_label['Q12'].values
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

rep_load_list = []
for i, home_idx in enumerate(unique_home_arr):
    distance = distance_list[i].mean(axis=0)
    mean_ = distance.mean()
    std_ = np.sqrt(distance.var())
    data = data_raw[home_arr == home_idx,:]

    idx = distance <= mean_ - std_
    
    if len(idx) <= 5:
        idx[:] = True
    if idx.sum() <= 5:
        idx[:] = True

    data = data[idx,:]
    rep_load_list.append(data.mean(axis=0))

rep_load_dict['CER_p'] = np.array(rep_load_list)

# %% 결과 저장
import pandas as pd
model_dict = dict()
history_dict = dict()
result_df = pd.DataFrame(columns = ['train_acc', 'val_acc', 'test_acc',\
    'train_auc', 'val_auc', 'test_auc', \
     'train_f1', 'val_f1', 'test_f1'])

# %% case split
# 0. training 10개 test 나머지 x 10번
# 1. training 20개 test 나머지 x 10번
# 2. training 30개 test 나머지 x 10번
# 3. training 40개 test 나머지 x 10번
# 4. training 50개 test 나머지 x 10번
# 5. training 100개 test 나머지 x 10번
# 6. training 200개 test 나머지 x 10번

np.random.seed(0)
train_dict = dict()
test_dict = dict()

for data_name in ['CER', 'SAVE']:
    n_sample = rep_load_dict[data_name].shape[0]
    for SEED in range(100):
        for case_idx, n_sample_train in enumerate([20, 30, 40, 50]):
            train_dict[f'{data_name}_case_{case_idx}_seed_{SEED}'] = np.random.randint(0, n_sample, n_sample_train)
            test_dict[f'{data_name}_case_{case_idx}_seed_{SEED}'] = np.array([i for i in range(n_sample) if i not in train_dict[f'{data_name}_case_{case_idx}_seed_{SEED}']])

# %% 1. self training

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
es = EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )

VAL_SPLIT = 0.25
for data_name in ['CER', 'SAVE']:
    for case_idx in [2, 3]:
        for SEED in range(10):
            model_name = f'{data_name}_self_case_{case_idx}_seed_{SEED}'
            label_ref = label_dict[data_name].copy()
            label_ref[label_ref<=2] = 0
            label_ref[label_ref>2] = 1

            data = rep_load_dict[data_name]
            params = make_param_int(params, ['batch_size'])
            label = to_categorical(label_ref.copy(), dtype=int)
            label = label.astype(float)

            train_idx = train_dict[f'{data_name}_case_{case_idx}_seed_{SEED}']
            test_idx = test_dict[f'{data_name}_case_{case_idx}_seed_{SEED}']
            X_train, y_train = data[train_idx], label[train_idx]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)
            X_test, y_test = data[test_idx], label[test_idx]

            model = DNN_model(params, True, label, 48)

            history_self = model.fit(X_train, y_train, epochs=params['epoch'], \
                    verbose=0, validation_data=(X_val, y_val),batch_size=params['batch_size'], callbacks = [es])
            model_dict[model_name] = model
            history_dict[model_name] = history_self

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
            val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
            test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)

            result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                        train_auc, val_auc, test_auc, \
                                        train_f1, val_f1, test_f1]

            print(f'data: {data_name} // case_idx: {case_idx} // seed: {SEED}')


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


# %% Transfer learning (source)
feature_dict = dict()
feature_dict['SAVE_mrmr'] = np.array([3, 9, 4])
feature_dict['SAVE_all'] = np.arange(0, 48)

feature_dict['CER_mrmr'] = np.array([1, 32])
feature_dict['CER_all'] = np.arange(0, 48)
feature_dict['CER_T6'] = range(36, 42) 

for feature_name in feature_dict.keys():
    if feature_name[0] == 'S':
        src_data_name = 'CER'
    else:
        src_data_name = 'SAVE'

    time_ = np.array(feature_dict[feature_name])

    valid_feature = np.zeros((48), dtype = bool)
    valid_feature[time_] = True

    model_name = f'{feature_name}_src'
    print(f'data:: {src_data_name}')
        
    label_ref = label_dict[src_data_name].copy()
    label_ref[label_ref<=2] = 0
    label_ref[label_ref>2] = 1

    # dataset filtering
    if src_data_name == 'SAVE':
        data = rep_load_dict['SAVE']
        data = data[filtered_idx_2,:]
        label_ref = label_ref[filtered_idx_2]
    else:
        data = rep_load_dict['CER']
        data = data[filtered_idx_1,:]
        label_ref = label_ref[filtered_idx_1]
        
    params = make_param_int(params, ['batch_size'])
    label = to_categorical(label_ref.copy(), dtype=int)
    label = label.astype(float)

    params = make_param_int(params, ['batch_size'])
    label = to_categorical(label_ref.copy(), dtype=int)
    label = label.astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = 0.9, random_state = 0, stratify = label)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0, stratify = y_train)

    model = DNN_model(params, True, label, valid_feature.sum())

    es = EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )

    history_self = model.fit(X_train, y_train, epochs=params['epoch'], \
            verbose=0, validation_data=(X_val, y_val),batch_size=params['batch_size'], callbacks = [es])
    model_dict[model_name] = model
    history_dict[model_name] = history_self

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
    val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
    test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)

    result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                    train_auc, val_auc, test_auc, \
                                    train_f1, val_f1, test_f1]


# %% Transfer learning (target)
import tensorflow as tf
for feature_name in ['CER_mrmr']:
    if feature_name[0] == 'S':
        src_data_name = 'CER'
        tgt_data_name = 'SAVE'
    else:
        src_data_name = 'SAVE'
        tgt_data_name = 'CER'

    for case_idx in [1]:
        for SEED in [1, 6]:
            for i in range(100):
                time_ = np.array(feature_dict[feature_name])
                valid_feature = np.zeros((48), dtype = bool)
                valid_feature[time_] = True

                model_name = f'{feature_name}_case_{case_idx}_seed_{SEED}'

                print(f'DATA:: {tgt_data_name}')
                    
                label_ref = label_dict[tgt_data_name].copy()
                label_ref[label_ref<=2] = 0
                label_ref[label_ref>2] = 1
                data = rep_load_dict[tgt_data_name][:,valid_feature]
                params = make_param_int(params, ['batch_size'])
                label = to_categorical(label_ref.copy(), dtype=int)
                label = label.astype(float)

                train_idx = train_dict[f'{data_name}_case_{case_idx}_seed_{SEED}']
                test_idx = test_dict[f'{data_name}_case_{case_idx}_seed_{SEED}']
                X_train, y_train = data[train_idx], label[train_idx]
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)
                X_test, y_test = data[test_idx], label[test_idx]

                # base_model = model_dict[f'{src_data_name}_src_timeset_{time_idx}']
                base_model = model_dict[f'{feature_name}_src']

                model = Sequential()
                for layer in base_model.layers[:-1]: # go through until last layer
                    model.add(layer)

                inputs = tf.keras.Input(shape=(valid_feature.sum(),))
                x = model(inputs, training=False)
                # x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
                x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
                model = tf.keras.Model(inputs, x_out)

                optimizer = Adam(params['lr'], epsilon=params['epsilon'])
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['acc'])
                es = EarlyStopping(
                                    monitor='val_loss',
                                    patience=10,
                                    verbose=0,
                                    mode='min',
                                    restore_best_weights=True
                                )
                history_tr_2 = model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_val, y_val),batch_size=params['batch_size'])

                model_dict[model_name] = model
                history_dict[model_name] = history_tr_2

                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)

                train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
                val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
                test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)
                if result_df.loc[model_name,'test_auc'] > test_auc:
                    result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                                train_auc, val_auc, test_auc, \
                                                train_f1, val_f1, test_f1]

# %%
result_df.to_csv('simulation_results/result_df_0701.csv')
for key, model in model_dict.items():
    model.save(f'model_save_0627/{key}')

# %%
type_ = 'test'
metric = 'auc'
data = 'CER'

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

valid_index = [data + '_all_case_' + str(case) for case in range(4)]
self_index = [data + '_self_case_' + str(case) for case in range(4)]
# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
# print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
# plt.title(f'{data} {type_} {metric} case 0')
plt.plot(result_df.loc[self_index,valid_col].values, marker = 's')
plt.plot(result_df.loc[valid_index,valid_col].values, marker = 's')
plt.xlabel('Time set')
plt.ylabel(metric.upper())
plt.legend(['Transfer learning','Self learning'])
plt.show()

# %% Evaluat1on
import seaborn as sns
plt.figure(figsize = (8, 5))
data_name = 'SAVE'
type_ = 'mrmr'
metric = 'auc'
for case_idx in [0,1]:
    result_1 = result_df.loc[f'{data_name}_self_case_{case_idx}_seed_0':f'{data_name}_self_case_{case_idx}_seed_9','test_'+metric]
    print(result_1)

    result_2 = result_df.loc[f'{data_name}_all_case_{case_idx}_seed_0':f'{data_name}_all_case_{case_idx}_seed_9','test_'+metric]
    print(result_2)

    result_3 = result_df.loc[f'{data_name}_{type_}_case_{case_idx}_seed_0':f'{data_name}_{type_}_case_{case_idx}_seed_9','test_'+metric]
    print(result_3)

    plt.subplot(2,1,case_idx+1)
    sns.boxplot(data = [result_1, result_2, result_3])
    plt.xticks(range(3), ['Self','Transfer','Proposed'])
    plt.ylim(0.2, 0.8)
    plt.ylabel('AUC')
plt.show()

# %%
result_1 = result_df.loc[f'{data_name}_self_case_0_seed_0':f'{data_name}_self_case_0_seed_9','test_'+metric]
print(result_1)

result_2 = result_df.loc[f'{data_name}_all_case_1_seed_0':f'{data_name}_all_case_1_seed_9','test_'+metric]
print(result_2)

import seaborn as sns
sns.boxplot(data = [result_1, result_2])
plt.xticks(range(2), ['case 1','case 2'])
plt.ylim(0.2, 0.8)
plt.show()
