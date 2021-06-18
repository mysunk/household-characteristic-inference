# %% 데이터 로드 및 전처리


import os
os.chdir('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from module.util_main import downsampling, dim_reduct
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


# %% 결과 저장
import pandas as pd
model_dict = dict()
history_dict = dict()
result_df = pd.DataFrame(columns = ['train_acc', 'val_acc', 'test_acc',\
    'train_auc', 'val_auc', 'test_auc', \
     'train_f1', 'val_f1', 'test_f1'])

result_df = pd.read_csv('simulation_results/result.csv', index_col = 0)

# %% benchmark testing (case에 따른 self training)
params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
case = [0.9, 0.95, 0.975, 0.9875, 0.99, 0.995, 0.998, 0.5]
VAL_SPLIT = 0.25

for data_name in ['SAVE', 'CER']:
    for case_idx in [6]:
        TEST_SPLIT = case[case_idx]
        model_name = data_name +'_self_case_'+str(case_idx)

        print(f'DATA:: {data_name}')
            
        label_ref = label_dict[data_name].copy()
        label_ref[label_ref<=2] = 0
        label_ref[label_ref>2] = 1

        data = rep_load_dict[data_name]
        params = make_param_int(params, ['batch_size'])
        label = to_categorical(label_ref.copy(), dtype=int)
        label = label.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

        model = DNN_model(params, True, label, 48)

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

        plot_history_v2([(model_name, history_self)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

        result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                    train_auc, val_auc, test_auc, \
                                    train_f1, val_f1, test_f1]

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
plt.ylabel('KL Divergence')
plt.title('KL divergence btw CER and SAVE')
plt.plot(kl_result,'.')
plt.hlines(ref, 0, n_data, color = 'r',zorder=5)
plt.xlabel('House idx')
plt.show()


#  SAVE 하나씩 제거해가며 CER과 비교

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

filtered_idx_1 = np.array(kl_result) > ref
filtered_idx_2 = np.array(kl_result_2) > ref_2

# %% source training (CER, SAVE)
p0 = range(0, 24*2)
p1 = range(0, 5*2)
p2 = range(5*2, 9*2)
p3 = range(9*2, 12*2)
p4 = range(12*2, 15*2)
p5 = range(15*2, 18*2)
p6 = range(18*2, 21*2)
p7 = range(21*2, 24*2)
p8 = range(0, 18)

time_list = [p0, p1, p2, p3, p4, p5, p6, p7, p8]

VAL_SPLIT = 0.25
TEST_SPLIT = 0.2

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}

for data_name in ['SAVE','CER']:
    for time_idx, time_ in enumerate(time_list[9:]):
        # time_idx += 9
        # K = (time_idx - 9) * 4
        # time_ = np.array(feature_dict[f'{data_name}_mrmr_K_{K}'])

        valid_feature = np.zeros((48), dtype = bool)
        valid_feature[time_] = True

        # model_name = f'{data_name}_src_timeset_{time_idx}'
        model_name = f'{data_name}_src_2_timeset_{time_idx}'
        print(f'DATA:: {data_name}')
            
        label_ref = label_dict[data_name].copy()
        label_ref[label_ref<=2] = 0
        label_ref[label_ref>2] = 1

        data = rep_load_dict[data_name]
        # dataset filtering
        if data_name == 'SAVE':
            data = data[filtered_idx_2,:]
            label_ref = label_ref[filtered_idx_2]
        else:
            data = data[filtered_idx_1,:]
            label_ref = label_ref[filtered_idx_1]
            
        params = make_param_int(params, ['batch_size'])
        label = to_categorical(label_ref.copy(), dtype=int)
        label = label.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

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

        plot_history_v2([(model_name, history_self)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

        result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                        train_auc, val_auc, test_auc, \
                                        train_f1, val_f1, test_f1]
# %% transfer learning with various time set
import tensorflow as tf

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
case = [0.9, 0.95, 0.975, 0.9875, 0.99, 0.995, 0.998, 0.5]
VAL_SPLIT = 0.25

for src_data_name, tgt_data_name in zip(['CER'], ['SAVE']):
    for case_idx in range(8):
        for time_idx, time_ in enumerate(time_list):
            time_idx = 1
            time_ = p1

            valid_feature = np.zeros((48), dtype = bool)
            valid_feature[time_] = True

            TEST_SPLIT = case[case_idx]
            model_name = f'{tgt_data_name}_trans_2_timeset_{time_idx}_case_{case_idx}'

            print(f'DATA:: {tgt_data_name}')
                
            label_ref = label_dict[tgt_data_name].copy()
            label_ref[label_ref<=2] = 0
            label_ref[label_ref>2] = 1

            data = rep_load_dict[tgt_data_name]
            params = make_param_int(params, ['batch_size'])
            label = to_categorical(label_ref.copy(), dtype=int)
            label = label.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

            # base_model = model_dict[f'{src_data_name}_src_timeset_{time_idx}']
            base_model = model_dict[f'{src_data_name}_src_2_timeset_{time_idx}']

            model = Sequential()
            for layer in base_model.layers[:-1]: # go through until last layer
                model.add(layer)

            inputs = tf.keras.Input(shape=(valid_feature.sum(),))
            x = model(inputs, training=False)
            # x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
            x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
            model = tf.keras.Model(inputs, x_out)

            optimizer = Adam(params['lr'] * 0.1, epsilon=params['epsilon'])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['acc'])
            es = EarlyStopping(
                                monitor='val_loss',
                                patience=100,
                                verbose=0,
                                mode='min',
                                restore_best_weights=True
                            )
            history_tr_2 = model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_val, y_val),batch_size=params['batch_size'])

            model_dict[model_name] = model
            history_dict[model_name] = history_self

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
    
            train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
            val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
            test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)

            plot_history_v2([(model_name, history_self)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

            result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                        train_auc, val_auc, test_auc, \
                                        train_f1, val_f1, test_f1]
            break

# %% source training with different feature selection methods
VAL_SPLIT = 0.25
TEST_SPLIT = 0.2

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}

for feature_name in feature_dict.keys():
    
    if feature_name[0] == 'S':
        data_name = 'CER'
    else:
        data_name = 'SAVE'

    time_ = np.array(feature_dict[feature_name])

    valid_feature = np.zeros((48), dtype = bool)
    valid_feature[time_] = True

    # model_name = f'{data_name}_src_timeset_{time_idx}'
    model_name = f'{feature_name}_src_2'
    print(f'DATA:: {data_name}')
        
    label_ref = label_dict[data_name].copy()
    label_ref[label_ref<=2] = 0
    label_ref[label_ref>2] = 1

    data = rep_load_dict[data_name]
    # dataset filtering
    if data_name == 'SAVE':
        data = data[filtered_idx_2,:]
        label_ref = label_ref[filtered_idx_2]
    else:
        data = data[filtered_idx_1,:]
        label_ref = label_ref[filtered_idx_1]
        
    params = make_param_int(params, ['batch_size'])
    label = to_categorical(label_ref.copy(), dtype=int)
    label = label.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

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

    plot_history_v2([(model_name, history_self)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

    result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                    train_auc, val_auc, test_auc, \
                                    train_f1, val_f1, test_f1]

# %% transfer learning with mrmr
import tensorflow as tf
params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
case = [0.9, 0.95, 0.975, 0.9875, 0.99, 0.995, 0.998, 0.5]
VAL_SPLIT = 0.25

for feature_name in feature_dict.keys():
    if feature_name[0] == 'S':
        src_data_name = 'CER'
        tgt_data_name = 'SAVE'
    else:
        continue
        src_data_name = 'SAVE'
        tgt_data_name = 'CER'

    for case_idx in range(8):
        time_ = np.array(feature_dict[feature_name])
        valid_feature = np.zeros((48), dtype = bool)
        valid_feature[time_] = True

        TEST_SPLIT = case[case_idx]
        model_name = f'{feature_name}_case_{case_idx}'

        print(f'DATA:: {tgt_data_name}')
            
        label_ref = label_dict[tgt_data_name].copy()
        label_ref[label_ref<=2] = 0
        label_ref[label_ref>2] = 1

        data = rep_load_dict[tgt_data_name]
        params = make_param_int(params, ['batch_size'])
        label = to_categorical(label_ref.copy(), dtype=int)
        label = label.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

        # base_model = model_dict[f'{src_data_name}_src_timeset_{time_idx}']
        base_model = model_dict[f'{feature_name}_src_2']

        model = Sequential()
        for layer in base_model.layers[:-1]: # go through until last layer
            model.add(layer)

        inputs = tf.keras.Input(shape=(valid_feature.sum(),))
        x = model(inputs, training=False)
        # x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
        x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
        model = tf.keras.Model(inputs, x_out)

        optimizer = Adam(params['lr'] * 0.1, epsilon=params['epsilon'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['acc'])
        es = EarlyStopping(
                            monitor='val_loss',
                            patience=100,
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

        plot_history_v2([(model_name, history_tr_2)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

        result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                    train_auc, val_auc, test_auc, \
                                    train_f1, val_f1, test_f1]

# %% only inference for save
tgt_data_name = 'SAVE'
# result_df_2 = pd.DataFrame(columns = ['train_acc', 'val_acc', 'test_acc',\
#     'train_auc', 'val_auc', 'test_auc', \
#      'train_f1', 'val_f1', 'test_f1'])

label_ref = label_dict[tgt_data_name].copy()
label_ref[label_ref<=2] = 0
label_ref[label_ref>2] = 1

data = rep_load_dict[tgt_data_name]
params = make_param_int(params, ['batch_size'])
label = to_categorical(label_ref.copy(), dtype=int)
label = label.astype(float)

for case_idx in tqdm(range(8)):
    TEST_SPLIT = case[case_idx]
    for i in range(1, 6):
        key = f'SAVE_mrmr_K_{i}_case_{case_idx}'
        time_ = np.array(feature_dict[f'SAVE_mrmr_K_{i}'])
        valid_feature = np.zeros((48), dtype = bool)
        valid_feature[time_] = True
        X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 1, stratify = label)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 1, stratify = y_train)

        model = model_dict[key]
        print(X_train.shape[0])

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
        val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
        test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)
        
        result_df_2.loc[key,:] = [train_acc, val_acc, test_acc,\
                                    train_auc, val_auc, test_auc, \
                                    train_f1, val_f1, test_f1]


# %% random state 바꿔서 비교
# %% 시각화 2
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

# plt.figure(figsize = (10, 5))
plt.title(f'{data} dataset {metric.upper()} result for different K')

results, results_2 = [], []
for time in range(1, 6, 1):
    valid_index = [data + f'_mrmr_K_{time}_'+'case_' + str(case) for case in range(7)]
    results.append(result_df.loc[valid_index,valid_col].values.reshape(-1))
    results_2.append(result_df_2.loc[valid_index,valid_col].values.reshape(-1))

import seaborn as sns
sns.boxplot(data = results)
plt.xlabel('K')
# plt.xticks(range(46)[::3], range(1, 48, 1)[::3])
plt.ylabel(metric.upper())
plt.show()

sns.boxplot(data = results_2)
plt.xlabel('K')
# plt.xticks(range(46)[::3], range(1, 48, 1)[::3])
plt.ylabel(metric.upper())
plt.show()


# %% source 자체 학습 결과는 제외
src_index =  [i for i in result_df.index if 'src' in i]
src_result_df = result_df.loc[src_index,:]
result_df.drop(index = src_index, inplace = True)

# %% train 부분 제외
# result_df_copy = result_df.copy()

drop_col =  [i for i in result_df.columns if 'val' in i]
result_df.drop(columns = drop_col, inplace = True)


# %% 시각화 2
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]


# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
# print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
plt.title(f'{data} dataset {metric.upper()} result for each time set')

results = np.zeros((8, 9))
for case in range(8):
    valid_index = [data + '_trans_2_timeset_' + str(i) + '_case_' + str(case) for i in range(9)]
    self_index = data + '_self_case_' + str(case)
    results[case,:] = result_df.loc[valid_index,valid_col].values.reshape(-1)

import seaborn as sns
sns.boxplot(data = results)
# plt.plot(result_df.loc[valid_index,valid_col].values)
# plt.hlines(result_df.loc[self_index, valid_col], 0, len(valid_index)-1, color = 'r',zorder=5,linestyles = '--')
plt.xlabel('Time set')
plt.xticks(range(len(valid_index)), ['T' +str(i) for i in range(9)])
plt.ylabel(metric.upper())
# plt.legend(['Transfer learning','Self learning'])
plt.ylim([0.55, 0.75])
plt.show()



# %% 시각화 2
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]


# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
# print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
plt.title(f'{data} dataset {metric.upper()} result for different K')

results = []
# self_index = data + '_self_case_' + str(case)
# results.append(result_df.loc[self_index,valid_col].values.reshape(-1))

for time in range(1, 48, 1):
    valid_index = [data + f'_mrmr_K_{time}_'+'case_' + str(case) for case in range(7)]
    results.append(result_df.loc[valid_index,valid_col].values.reshape(-1))

import seaborn as sns
sns.boxplot(data = results)
# plt.plot(result_df.loc[valid_index,valid_col].values)
# plt.hlines(result_df.loc[self_index, valid_col], 0, len(valid_index)-1, color = 'r',zorder=5,linestyles = '--')
plt.xlabel('K')
plt.xticks(range(46)[::3], range(1, 48, 1)[::3])
# plt.xticks(range(len(valid_index)), ['T' +str(i) for i in range(9)])
plt.ylabel(metric.upper())
# plt.legend(['Transfer learning','Self learning'])
plt.ylim([0.4, 0.76])
plt.show()

# %% 시각화 3
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]


# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
# print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
plt.title(f'{data} dataset {metric.upper()} result for each time set')

plt.plot(results[:3,:].T)
# plt.plot(result_df.loc[valid_index,valid_col].values)
# plt.hlines(result_df.loc[self_index, valid_col], 0, len(valid_index)-1, color = 'r',zorder=5,linestyles = '--')
plt.xlabel('Time set')
plt.xticks(range(len(valid_index)), ['T' +str(i) for i in range(9)])
plt.ylabel(metric.upper())
# plt.legend(['Transfer learning','Self learning'])
# plt.ylim([0.55, 0.75])
plt.show()


# %% 시각화 어떻게할까..
case = 4
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

valid_index = [data + '_trans_2_timeset_' + str(i) + '_case_' + str(case) for i in range(8)]
self_index = data + '_self_case_' + str(case)
# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
# print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
plt.title(f'{data} {type_} {metric} case 0')
plt.plot(result_df.loc[valid_index,valid_col].values, marker = 's')
plt.hlines(result_df.loc[self_index, valid_col], 0, len(valid_index)-1, color = 'r',zorder=5,linestyles = '--')
plt.xlabel('Time set')
plt.xticks(range(len(valid_index)), ['T' +str(i) for i in range(8)])
plt.ylabel(metric.upper())
plt.legend(['Transfer learning','Self learning'])
# plt.ylim([0.66, 0.7])
plt.show()

# %% case에 따라 transfer learning과 self learning 비교
metric = 'auc'
data = 'SAVE'
valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

valid_index_1 = [data + '_trans_2_timeset_0_case_' + str(case) for case in [7,0,1,2,3,4,5]]

valid_index_4 = [data + '_trans_2_timeset_2_case_' + str(case) for case in [7,0,1,2,3,4,5]]

valid_index_3 = [data + f'_mrmr_K_3_'+'case_' + str(case) for case in [7,0,1,2,3,4,5]]

# valid_index_3 = [data + '_trans_2_timeset_2_case_' + str(case) for case in [7,0,1,2,3,4,5,6]]
# valid_index_4 = [data + '_trans_2_timeset_1_case_' + str(case) for case in [7,0,1,2,3,4,5,6]]
valid_index_2 = [data + '_self_case_' + str(case) for case in [7,0,1,2,3,4,5]]

plt.figure(figsize = (10, 5))
plt.plot(result_df.loc[valid_index_1, valid_col].values, label = 'Transfer learning', marker = 's')
plt.plot(result_df.loc[valid_index_3, valid_col].values, label = 'Transfer learning + mRMR', marker = 'd')
plt.plot(result_df.loc[valid_index_4, valid_col].values, label = 'Transfer learning + T2', marker = '<')

# plt.plot(result_df.loc[valid_index_4, valid_col].values, label = 'Transfer learning for T1', marker = 'd')
plt.plot(result_df.loc[valid_index_2, valid_col].values, label = 'Self learning', marker = '^')
plt.title(f'{data} {type_} {metric.upper()}')
plt.legend()
plt.xlabel('Case idx')
plt.ylabel(metric.upper())
plt.ylim(0.6, 0.79)
plt.show()

# %%
print(feature_dict['CER_mrmr_K_4'])
print(feature_dict['SAVE_mrmr_K_16'])



# %% 결과 확인
case = 'case_6'
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_index = [i for i in result_df.index if data in i]
valid_index = [i for i in valid_index if case in i]

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]
valid_index = np.sort(valid_index)
# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
plt.title(f'{data} {type_} {metric} {case}')
plt.plot(result_df.loc[valid_index,valid_col], marker = 's')
plt.xticks(rotation = 30)
plt.legend(valid_col)
# plt.ylim([0.66, 0.7])
plt.show()

# %% trans 1과 2 비교
case = 'case_6'
type_ = 'test'
metric = 'auc'
data = 'SAVE'

valid_index = [i for i in result_df.index if data in i]
valid_index = [i for i in valid_index if case in i]

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

valid_index = np.sort(valid_index)
# print(result_df.loc[valid_index,valid_col].sort_values(by=f'{type_}_{metric}', ascending=False))
print(result_df.loc[valid_index,valid_col])

plt.figure(figsize = (10, 5))
plt.title(f'{data} {type_} {metric} {case}')
# plt.plot(result_df.loc[valid_index_1,valid_col].values, marker = 's', label = 'all')
plt.plot(result_df.loc[valid_index,valid_col], marker = 's')
plt.xticks(rotation = 30)
# plt.legend()
plt.legend(valid_col)
# plt.ylim([0.66, 0.7])
plt.show()

# %% Negative transfer 관련
data = 'CER'
metric = 'acc'

index_1 = [ '_trans_timeset_0_case_7','_trans_timeset_0_case_0',
 '_trans_timeset_0_case_1',
 '_trans_timeset_0_case_2',
 '_trans_timeset_0_case_3',
 '_trans_timeset_0_case_4',
 '_trans_timeset_0_case_5',
 '_trans_timeset_0_case_6',
]

index_2 = ['_trans_2_timeset_0_case_7', '_trans_2_timeset_0_case_0',
 '_trans_2_timeset_0_case_1',
 '_trans_2_timeset_0_case_2',
 '_trans_2_timeset_0_case_3',
 '_trans_2_timeset_0_case_4',
 '_trans_2_timeset_0_case_5',
 '_trans_2_timeset_0_case_6'
 ]

index_1 = [data + text for text in index_1]
index_2 = [data + text for text in index_2]

valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

plt.figure(figsize = (10, 5))
plt.title(f'{data} dataset ' + metric.upper())

plt.plot(result_df.loc[index_1,valid_col].values, marker = 's')
plt.plot(result_df.loc[index_2,valid_col].values, marker = 's')

plt.xticks()
plt.legend(['all', 'proposed'])
plt.xlabel('Case index')
plt.ylabel(metric.upper())
# plt.ylim([0.76, 0.8])
plt.show()

# %% case별 확인
type_ = 'test'
metric = 'auc'
data = 'CER'

valid_index = [i for i in result_df.index if data in i]

valid_index = [i for i in valid_index if 'self' in i]
# valid_index = [i for i in valid_index if len(i) == 26]
valid_col = [i for i in result_df.columns if metric in i]
valid_col = [i for i in valid_col if type_ in i]

plt.figure(figsize = (10, 5))
plt.title(f'{data} {type_} {metric}')
plt.plot(result_df.loc[valid_index,valid_col], marker = 's')
plt.xticks(rotation = 30)
plt.legend(valid_col)
# plt.ylim([0.76, 0.8])
plt.show()
print(result_df.loc[valid_index,valid_col])

# %% Distribution 차이 보기
cer_label = label_dict['CER']
cer_data = rep_load_dict['CER']

save_label = label_dict['SAVE']
save_data = rep_load_dict['SAVE']

plt.figure(figsize = (6, 3))
plt.hist(cer_data.reshape(-1), bins = 100, label = 'CER', alpha = 0.3, density = True)
# plt.hist(cer_data[cer_label > 2,:10].reshape(-1), bins = 100, label = 'CER', alpha = 0.3, density = True)
plt.hist(save_data.reshape(-1), bins = 100, label = 'SAVE', alpha = 0.3, density = True)
# plt.hist(save_data[save_label > 2,:10].reshape(-1), bins = 100, label = 'SAVE', alpha = 0.3, density = True)
plt.xlabel('Energy')
plt.ylabel('Density')
plt.title('Distributions')
plt.xlim(0, 3)
plt.legend()
plt.show()

# %% dataset distance
from scipy.spatial.distance import cosine
tmp_result , tmp_result_2, tmp_result_3 = [], [], []
view = 14

for i in range(cer_data.shape[0]):
    tmp_result.append(cosine(cer_data[view,:], cer_data[i,:]))

for i in range(save_data.shape[0]):
    tmp_result_2.append(cosine(cer_data[view,:], save_data[i,:]))
plt.plot(tmp_result)
plt.plot(tmp_result_2)
plt.show()

# %%
tmp_result , tmp_result_2, tmp_result_3 = [], [], []
view = 8

for i in range(save_data.shape[0]):
    tmp_result.append(cosine(save_data[view,:], save_data[i,:]))

for i in range(cer_data.shape[0]):
    tmp_result_2.append(cosine(save_data[view,:], cer_data[i,:]))
plt.plot(tmp_result)
plt.plot(tmp_result_2)
plt.show()

# %%
from sklearn.metrics import pairwise_distances
n_c = cer_data.shape[0]
n_s = save_data.shape[0]

dataset_distance_df = pd.DataFrame()

for metric in ['euclidean', 'manhattan', 'cosine', 'cityblock']:
    for i in range(6):
        cer_sim_arr = pairwise_distances(cer_data[:,i * 8:(i+1)*8], cer_data[:,i * 8:(i+1)*8], metric = metric)
        cross_sim_arr = pairwise_distances(cer_data[:,i * 8:(i+1)*8], save_data[:,i * 8:(i+1)*8], metric = metric)
        save_sim_arr = pairwise_distances(save_data[:,i * 8:(i+1)*8], save_data[:,i * 8:(i+1)*8], metric = metric)
        print('{:.2f}, {:.2f}, {:.2f}'.format(cer_sim_arr.mean(), save_sim_arr.mean(), cross_sim_arr.mean()))

        dataset_distance_df.loc[i, metric + '_CER'] = cer_sim_arr.mean()
        dataset_distance_df.loc[i, metric + '_SAVE'] = save_sim_arr.mean()
        dataset_distance_df.loc[i, metric + '_CROSS'] = cross_sim_arr.mean()

# %% target: cer, source: save일 때
threshold = 1.1
sim_result = np.zeros((n_c), dtype = bool)
for view in range(n_c):
    val_1 = cer_sim_arr[view,:].mean()
    val_2 = cross_sim_arr[:,view].mean()
    if val_2 < val_1 * threshold:
        sim_result[view] = True
print(sim_result.sum())

# %%
# distribution 차이 확인
plt.hist(cer_data.mean(axis=1), bins = 100, label = 'CER', alpha = 0.3, density = True)
plt.hist(save_data.mean(axis=1), bins = 100, label = 'SAVE', alpha = 0.3, density = True)
plt.legend()
plt.show()

# %%
# plt.plot(cer_data[cer_label == 1,:].mean(axis=0), label = 'CER')
plt.plot(save_data[save_label == 1,:].mean(axis=0), label = 'SAVE')
plt.plot(save_data[save_label == 2,:].mean(axis=0), label = 'SAVE')
plt.plot(save_data[save_label == 3,:].mean(axis=0), label = 'SAVE')
plt.legend()
plt.show()

# %%
plt.plot(cer_data[cer_label == 1,:].mean(axis=0))
plt.plot(cer_data[cer_label == 2,:].mean(axis=0))
plt.plot(cer_data[cer_label == 3,:].mean(axis=0))
plt.legend()
plt.show()


# %%
view = 9
view2 = np.argmin(cross_sim_arr[view,:]) 
plt.plot(save_data[view,:], label = 'SAVE')
# plt.plot(cer_data[np.argmax(cross_sim_arr[view,:]),:], label = 'not sim')
plt.plot(cer_data[view2,:], label = 'CER')
plt.title(f'save: {int(save_label[view])} cer: and {cer_label[view2]}')
plt.legend()
plt.show()


# %%
v, c = np.unique(cer_label, return_counts = True)
plt.plot(v, c / c.sum())

v, c = np.unique(save_label, return_counts = True)
plt.plot(v, c / c.sum())
plt.show()

# %%
view = 0
view2 = np.argsort(cer_sim_arr[view,:])[2]
plt.plot(cer_data[view,:], label = cer_label[view])
# plt.plot(cer_data[np.argmax(cross_sim_arr[view,:]),:], label = 'not sim')
plt.plot(cer_data[view2,:], label = cer_label[view2])
# plt.title(f'{} and {cer_label[view2]}')
plt.legend()
plt.show()


# %%
from scipy.spatial.distance import cityblock
euc_dist = np.linalg.norm(A-B)
man_dist = cityblock(A, B)


# %% save
result_df.to_csv('simulation_results/result.csv', index = True)


# %% simulation 2: different pre-train method
case = [0.9, 0.95, 0.975, 0.9875, 0.99, 0.995, 0.999]
VAL_SPLIT = 0.2
for src_data_name, tgt_data_name in zip(['SAVE','CER'], ['CER','SAVE']):
    for case_idx in [4, 5, 6]:
        for time_idx, time_ in enumerate(time_list):
            valid_feature = np.zeros((48), dtype = bool)
            valid_feature[time_] = True

            TEST_SPLIT = case[case_idx]
            model_name = f'{tgt_data_name}_trans_timeset_{time_idx}_case_{case_idx}_2'

            print(f'DATA:: {tgt_data_name}')
                
            label_ref = label_dict[tgt_data_name].copy()
            label_ref[label_ref<=2] = 0
            label_ref[label_ref>2] = 1

            data = rep_load_dict[tgt_data_name]
            params = make_param_int(params, ['batch_size'])
            label = to_categorical(label_ref.copy(), dtype=int)
            label = label.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

            base_model = model_dict[f'{src_data_name}_src_timeset_{time_idx}']

            model = Sequential()
            for layer in base_model.layers[:-1]: # go through until last layer
                layer.trainable = False
                model.add(layer)
            # train only the last layer
            layer = base_model.layers[-1]
            layer.trainable = True
            model.add(layer)

            # inputs = tf.keras.Input(shape=(valid_feature.sum(),))
            # x = model(inputs, training=True)
            # # x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
            # x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
            # model = tf.keras.Model(inputs, x_out)

            optimizer = Adam(params['lr'], epsilon=params['epsilon'])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['acc'])
            es = EarlyStopping(
                                monitor='val_loss',
                                patience=100,
                                verbose=0,
                                mode='min',
                                restore_best_weights=True
                            )
            history_tr_2 = model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_val, y_val),batch_size=params['batch_size'])

            model_dict[model_name] = model
            history_dict[model_name] = history_self

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
    
            train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
            val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
            test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)

            plot_history_v2([(model_name, history_self)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

            result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
                                        train_auc, val_auc, test_auc, \
                                        train_f1, val_f1, test_f1]

# %% mrmr
from mrmr import mrmr_classif
from tqdm import tqdm

feature_dict = dict()

for data_name in ['SAVE', 'CER']:
    for K in tqdm(range(4, 48, 4)):
        X = rep_load_dict[data_name]
        y = label_dict[data_name].copy()
        y[y<=2] = 0
        y[y>2] = 1

        X = pd.DataFrame(X)
        y = pd.Series(y)

        selected_features = mrmr_classif(X, y, K = K)
        feature_dict[f'{data_name}_mrmr_K_{K}'] = selected_features

# %% mrmr simulation 2
from mrmr import mrmr_classif
from tqdm import tqdm

# feature_dict = dict()

for data_name in ['CER']:
    for K in tqdm([47]):
        data = rep_load_dict[data_name]
        label = label_dict[data_name].copy()
        label[label<=2] = 0
        label[label>2] = 1
        
        TEST_SPLIT = case[-1]
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = TEST_SPLIT, random_state = 0, stratify = label)

        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)

        selected_features = mrmr_classif(X_train, y_train, K = K)
        feature_dict[f'{data_name}_mrmr_K_{K}'] = selected_features

# %%
plt.figure(figsize = (6, 3))
plt.plot(CER.iloc[:48*7,0].values, label = 'CER')
plt.plot(SAVE.iloc[:48*7,2].values, label = 'SAVE')
plt.xticks([])
plt.ylabel('Energy')
plt.xlabel('Date')
plt.legend()
plt.show()

# %%
data_1 = CER.values[:,1].reshape(-1)
data_1 = data_1[~pd.isnull(data_1)]
data_2 = SAVE.values[:,1].reshape(-1)
data_2 = data_2[~pd.isnull(data_2)]

# plt.hist(data_1, density = True, bins = 30, alpha = 0.5)
plt.hist(data_2, density = True, bins = 100, alpha = 0.5)
plt.show()

# %% feature dict 분석
feature_dict.keys()



# %%
feature_dict['SAVE_mrmr_K_47']

# %%
for key, model in model_dict.items():
    model.save(f'model_save/{key}')
