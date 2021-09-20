'''
@ paper:
Wang, Yi, et al. 
"Deep learning-based socio-demographic information identification from smart meter data."
IEEE Transactions on Smart Grid 10.3 (2018): 2593-2602.
'''

# %% load dataset
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
font = {'size': 16}
matplotlib.rc('font', **font)

from pathlib import Path
import pandas as pd


### 에너지 데이터 로드

# CER 데이터 로드
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

start_date = pd.to_datetime('2010-01-01 00:00:00')
end_date = pd.to_datetime('2010-06-30 23:30:00')

CER = CER.loc[start_date:end_date,:]

# invalid house processing
nan_ratio = pd.isnull(CER).sum(axis=0) / CER.shape[0]
invalid_idx = (nan_ratio == 1)
CER = CER.loc[:,~invalid_idx]
CER_label = CER_label.loc[~invalid_idx,:]

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

start_date = pd.to_datetime('2018-01-01 00:00:00')
end_date = pd.to_datetime('2018-06-30 23:45:00')

SAVE = SAVE.loc[start_date:end_date,:]

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

print(CER.shape)
print(CER_label.shape)

print(SAVE.shape)
print(CER.shape)

# %% functions
from sklearn.model_selection import train_test_split

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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K

def svm_loss(layer, C):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = C * K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        regularization_loss = tf.reduce_mean(tf.square(weights_tf))
        return regularization_loss + hinge_loss

    return categorical_hinge_loss
def CNN_svm(params, binary, label):
    x_input = Input(shape=(7, 48, 1))
    x = Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 48, 1))(x_input)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(32, input_shape=(320,))(x)

    # Add svm layer
    x_out = Dense(label.shape[1], input_shape=(32,),
       use_bias=False, activation='relu', name='svm')(x)

    model = Model(x_input, x_out)
    optimizer = Adam(params['lr'], epsilon=params['epsilon'])
    # optimizer = SGD(lr=params['lr'], decay=0.01, momentum=0.9)
    # optimizer = tf.keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)

    
    # model.compile(optimizer=optimizer, loss=svm_loss(model.get_layer('svm'), params['C']),
    # metrics = ['acc'])
    model.compile(optimizer=optimizer, loss='hinge', metrics = ['acc'])
    return model

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

from sklearn.model_selection import train_test_split
from sklearn import metrics
case = [0.5, 0.9, 0.95, 0.975, 0.9875, 0.99, 0.995]
VAL_SPLIT = 0.25

def evaluate(y_true, y_pred):
        '''
        return acc, auc, f1 score
        '''
        acc_ = (np.argmax(y_pred, axis=1) == y_true).mean()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:,0], pos_label=0)
        auc_ = metrics.auc(fpr, tpr)
        f1_score_ = metrics.f1_score(y_true, np.argmax(y_pred, axis=1))
        print('accuracy: {:.3f}, auc: {:.3f}, f1 score: {:.3f}'.format(acc_, auc_, f1_score_))
        return acc_, f1_score_, auc_



# %% simulation
params = {
            'lr': 0.1,
            'epoch': 1000,
            'batch_size': 128,
            'lambda': 0.01,
            'epsilon': 0.01,
            'C': 0.1
        }
results_benc_2 = []
for SEED in range(10):
    result_df = pd.DataFrame(columns = ['AC','F1', 'AUC'])
    for data_idx, data_name in enumerate(['CER', 'SAVE']):
        if data_name == 'CER':
            data_df = CER
            tmp_labels = CER_label['Q13'].values.copy()
        elif data_name == 'SAVE':
            data_df = SAVE
            tmp_labels = SAVE_label['Q2'].values.copy()
        
        invalid_idx = pd.isnull(tmp_labels)
        data_df = data_df.iloc[:,~invalid_idx]
        tmp_labels = tmp_labels[~invalid_idx]

        tmp_cols = range(data_df.shape[1])
        tmp_labels[tmp_labels<3] = 0
        tmp_labels[tmp_labels>2] = 1

        for case_idx in range(len(case)):
            TEST_SPLIT = case[case_idx]

            X_train_idx, X_test_idx, y_train, y_test = train_test_split(tmp_cols, tmp_labels, test_size = TEST_SPLIT, random_state = SEED, stratify = tmp_labels)
            X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_train_idx, y_train, test_size = VAL_SPLIT, random_state = SEED, stratify = y_train)
            data_rs_train, home_idx_c_train = transform(data_df.iloc[:,X_train_idx], 24 * 2 * 7)
            label_train = np.array([y_train[idx] for idx in home_idx_c_train])
            data_rs_val, home_idx_c_val = transform(data_df.iloc[:,X_val_idx], 24 * 2 * 7)
            label_val = np.array([y_val[idx] for idx in home_idx_c_val])

            # test는 augmentation x
            data_rs_test, home_idx_c_test = transform(data_df.iloc[:,X_test_idx], 24 * 2 * 7)
            unique_arr = np.unique(home_idx_c_test)
            data_list = []
            for idx in unique_arr:
                data_list.append(data_rs_test[home_idx_c_test == idx].mean(axis=0))
            X_test = np.array(data_list)
            label_test = np.array([y_test[idx] for idx in unique_arr])

            X_train, X_val, X_test = data_rs_train.reshape(-1, 7, 48, 1), data_rs_val.reshape(-1, 7, 48, 1), X_test.reshape(-1, 7, 48, 1)

            
            params = make_param_int(params, ['batch_size'])

            label_train_c = to_categorical(label_train.copy(), dtype=int).astype(float)
            label_val_c = to_categorical(label_val.copy(), dtype=int).astype(float)

            model = CNN_svm(params, True, label_train_c)
            es = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=0,
                    mode='min',
                    restore_best_weights=True
                )

            history = model.fit(X_train, label_train_c, epochs=params['epoch'], verbose=1, callbacks=[es], validation_data=(X_val, label_val_c),
                        batch_size=params['batch_size'])

            history_dict[data_name + str(case_idx)] = history

            y_pred = model.predict(X_test)
            result_df.loc[data_idx * len(case) + case_idx] = evaluate(label_test, y_pred)
            # print(result_df)
    results_benc_2.append(result_df)

# %% 결과 확인
def plot_history_v2(histories, save_path):

    fig, ax1 = plt.subplots(figsize = (10, 5))
    
    key = 'loss'
    for name, history in histories:
        plt.title(name)
        val = ax1.plot(history.epoch, history.history['val_'+key],
                    '--', label='val_loss', color = 'tab:blue')
        ax1.plot(history.epoch, history.history[key], \
                label='train_loss', color = 'tab:blue')
        
        idx = np.argmin(history.history['val_'+key
        best_tr = history.history[key][idx]
        best_val = history.history['val_'+key][idx]
    ax1.set_ylabel('Cross entropy loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axvline(x=idx, color = 'r')

    key = 'acc'
    ax2 = ax1.twinx()
    for name, history in histories:
        val = ax2.plot(history.epoch, history.history['val_'+key],
                    '--', label='val_acc', color = 'tab:orange')
        ax2.plot(history.epoch, history.history[key], \
                label= 'train_acc', color = 'tab:orange')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_xlabel('Epoch')

    fig.tight_layout()
    # print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr,key, best_val))
    plt.xlabel('Epochs')
    # plt.legend()
    # plt.xlim([0,max(history.epoch)])
    # plt.savefig(save_path, dpi = 100)
    plt.show()

plot_history_v2([('', history_dict['CER0'])], '')