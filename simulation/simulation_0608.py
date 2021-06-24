# %% load dataset & preprocessing
# SAVE 데이터 로드
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


# %% 필요 함수 정의
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model

def DNN_model(params, binary, label, n_feature):

    '''

    Parameters
    ----------
    params: DNN learning parameters
    binary: binary or not
    label: y

    Returns
    -------
    model: compiled DNN model

    '''

    x_input = Input(shape=(n_feature,))
    # x_input = Input(shape=(2, 24, 1))
    x = Dense(64, activation='relu', input_shape=(n_feature,))(x_input)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(64)(x)

    # Add svm layer
    x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)

    model = Model(x_input, x_out)
    optimizer = Adam(params['lr'], epsilon=params['epsilon'])
    # model.compile(optimizer=optimizer, loss='squared_hinge')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics =['acc'])
    return model

def plot_history(histories, key='loss'):
    plt.figure(figsize=(10,5))

    for name, history in histories:
      val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
      plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')
      
      if key == 'acc':
        idx = np.argmax(history.history['val_'+key])
      else:
        idx = np.argmin(history.history['val_'+key])
      best_tr = history.history[key][idx]
      best_val = history.history['val_'+key][idx]
      
      print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr,key, best_val))

    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param


# %% Simulation 결과 저장
model_dict = dict()
result_dict = dict()
history_dict = dict()

# %% Self learning (benchmark)
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn import metrics
import numpy as np
import scipy.io
import scipy.linalg
from sklearn.model_selection import train_test_split
params = {
    'lr': 0.001,
    'epoch': 1000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
for lr in [0.1, 0.01, 0.001, 0.0001]:
    params['lr'] = lr
    
    label_ref = label_dict['SAVE'].copy()
    label_ref[label_ref<=2] = 0
    label_ref[label_ref>2] = 1

    data = rep_load_dict['SAVE']
    params = make_param_int(params, ['batch_size'])
    label = to_categorical(label_ref.copy(), dtype=int)
    label = label.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.99, random_state = 0, stratify = label)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0, stratify = y_train)

    model = DNN_model(params, True, label, 48)

    es = EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )

    import tensorflow as tf
    y_preds = []
    class PredictionCallback(tf.keras.callbacks.Callback):
        def __init__(self, val):
            self.val = val

        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict(self.val)
            # save the result in each training
            y_preds.append(y_pred)

    history_self = model.fit(X_train, y_train, epochs=1000, verbose=0, validation_data=(X_val, y_val),batch_size=params['batch_size'] \
                        , callbacks = [es])
    model_dict['SAVE'] = model
    history_dict['SAVE_self'] = history_self

    y_pred = model.predict(X_test)
    result_test_self = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()
    fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_test, axis=1), y_pred[:,0], pos_label=0)
    auc_test_self = metrics.auc(fpr, tpr)

    y_pred = model.predict(X_val)
    result_val_self = (np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1)).mean()
    fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_val, axis=1), y_pred[:,0], pos_label=0)
    auc_val_self = metrics.auc(fpr, tpr)

    print(f'Validation acc is {result_val_self} and Test acc is {result_test_self}')
    print(f'Validation auc is {auc_val_self} and Test auc is {auc_test_self}')

    # plot_history([('self', history_self)],key='loss')
    # plot_history([('self', history_self)],key='acc')


# %% pca with prediction results
from sklearn.decomposition import PCA
plt.figure()
plt.title('Trajactory of prediction result')
pca = PCA(n_components=2)
n_epochs_valid = len(y_preds)
preds_tr = pca.fit_transform(np.array(y_preds).reshape(n_epochs_valid, -1))
GNT = y_val.reshape(1, -1)
GNT_tr = pca.transform(GNT)

plt.plot(preds_tr[:,0], preds_tr[:,1],'k.')
plt.plot(preds_tr[-1,0], preds_tr[-1,1],'gd')
plt.plot(GNT_tr[0,0], GNT_tr[0,1],'rd', label = 'GNT')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()

# %% proposed 2: corr이 평균 이상일 경우
corr_dict = dict()
for name in ['CER','SAVE']:
    # data_ex = feature_extraction_v2(rep_load_dict[name]).T
    # data_concat = np.concatenate([rep_load_dict[name], data_ex], axis=1)
    _, corr_, _ = evaluate_features(rep_load_dict[name], label_dict[name])
    corr_dict[name] = corr_
    # print(corr_)
same = corr_dict['SAVE'] > corr_dict['SAVE'].mean()

# %% feature sets
p0 = range(0, 24*2)
p1 = range(0, 5*2)
p2 = range(5*2, 9*2)
p3 = range(9*2, 12*2)
p4 = range(12*2, 15*2)
p5 = range(15*2, 18*2)
p6 = range(18*2, 21*2)
p7 = range(21*2, 24*2)
time_list = [p0, p1, p2, p3, p4, p5, p6, p7]

# %% Transfer leraning with various feature sets
from module.util import *
params = {
    'lr': 0.0001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
for time_idx, time_ in enumerate(time_list):
    time_idx = 1
    time_ = time_list[time_idx]
    for lr_idx, lr in enumerate([0.1, 0.01, 0.001, 0.0001]):

        params['lr'] = lr
        print(f'lr is {lr}')

        same = np.zeros((48), dtype = bool)
        same[time_] = True

        # model_name = 'CER_proposed1_lr_' + str(lr_idx)
        model_name = 'CER_case2_set' + str(time_idx)
        
        # model_name = 'CER_concat_lr_' + str(lr_idx)
        if lr_idx == 0:
            def scheduler(epoch, lr):
                if epoch < 30:
                    return lr
                else:
                    return lr * tf.math.exp(-0.1)

            label_ref = label_dict['CER'].copy()
            label_ref[label_ref<=2] = 0
            label_ref[label_ref>2] = 1

            # data = rep_load_dict['CER']
            data = rep_load_dict['CER']
            params = make_param_int(params, ['batch_size'])
            label = to_categorical(label_ref.copy(), dtype=int)
            label = label.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(data[:,same], label, test_size = 0.2, random_state = 0, stratify = label)

            """ 
            CNN based feature selection
            """
            # reshape
            # X_train = X_train.reshape(-1,1, same.sum(), 1)
            # X_test = X_test.reshape(-1,1, same.sum(), 1)

            # model = CNN_softmax(params, True, label)
            params_2 = params.copy()
            params_2['lr'] = 0.001
            model = DNN_model(params_2, True, label, n_feature = same.sum())
            # optimizer = Adam(params['lr'] * 0.1, epsilon=params['epsilon'])
            # model.compile(optimizer=optimizer, loss='binary_crossentropy')

            es = EarlyStopping(
                monitor='val_loss',
                patience=30,
                verbose=0,
                mode='min',
                restore_best_weights=True
            )
            lr =  tf.keras.callbacks.LearningRateScheduler(scheduler)

            history_tr_1 = model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, validation_data=(X_test, y_test), \
                                batch_size=params['batch_size'], callbacks = [es])
            model_dict[model_name] = model
            y_pred = model.predict(X_test)
            # print(y_pred)
            result = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()

        ####
        label_ref = label_dict['SAVE'].copy()
        label_ref[label_ref<=2] = 0
        label_ref[label_ref>2] = 1

        data = rep_load_dict['SAVE']
        params = make_param_int(params, ['batch_size'])
        label = to_categorical(label_ref.copy(), dtype=int)
        label = label.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(data[:,same], label, test_size = 0.975, random_state = 0, stratify = label)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0, stratify = y_train)

        """ 
        CNN based feature selection
        """
        # reshape
        # X_train = X_train.reshape(-1,7, 24*2, 1)
        # X_test = X_test.reshape(-1,7, 24*2, 1)

        base_model = model_dict[model_name]
        # base_model.trainable = False

        model = Sequential()
        for layer in base_model.layers[:-1]: # go through until last layer
            model.add(layer)

        inputs = tf.keras.Input(shape=(same.sum(),))
        x = model(inputs, training=False)
        # x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
        x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
        model = tf.keras.Model(inputs, x_out)

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

        y_pred = model.predict(X_test)
        result_test_tr = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()
        fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_test, axis=1), y_pred[:,0], pos_label=0)
        auc_test_tr = metrics.auc(fpr, tpr)

        y_pred = model.predict(X_val)
        result_val_tr = (np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1)).mean()
        fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_val, axis=1), y_pred[:,0], pos_label=0)
        auc_val_tr = metrics.auc(fpr, tpr)

        print(f'Validation acc is {result_val_tr} and Test acc is {result_test_tr}')
        print(f'Validation auc is {auc_val_tr} and Test auc is {auc_test_tr}')
        result_dict[model_name + '_val_acc'] = result_val_tr
        result_dict[model_name + '_val_auc'] = auc_val_tr
        result_dict[model_name + '_test_acc'] = result_test_tr
        result_dict[model_name + '_test_auc'] = auc_test_tr

        # plot_history([('transfer', history_tr_2)],key='loss')
        # plot_history([('transfer', history_tr_2)],key='acc')
        history_dict[model_name] = history_tr_2

# %% 결과 확인
for key, value in result_dict.items():
    if 'lr_4_val_acc' in key:
        # print(key)
        print(value)

# %% history plot
plot_history([('transfer', history_dict['CER_proposed1_lr_1'])],key='acc')

# %% feature extraction
from module.util_0607 import *
corr_dict = dict()
for name in ['CER','SAVE']:
    data_ex = feature_extraction_v2(rep_load_dict[name]).T
    data_concat = np.concatenate([rep_load_dict[name], data_ex], axis=1)
    rep_load_dict[name + '_concat'] = data_concat
    _, corr_, _ = evaluate_features(rep_load_dict[name], label_dict[name])
    corr_dict[name + '_concat'] = corr_
    # print(corr_)


# %% feature importance score
data = rep_load_dict['SAVE']
label = label_dict['SAVE'].copy()
label[label<=2] = 0
label[label>2] = 1

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.95, random_state = 0, stratify = label)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0, stratify = y_train)

rfc = RandomForestRegressor()
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)


# %%
for layer in model.layers:
    weights = layer.get_weights()
    break

# %%

base_model = model_dict[model_name]
for layer in base_model.layers:
    weights = layer.get_weights()
    print(weights)

# %%
