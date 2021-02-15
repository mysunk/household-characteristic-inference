import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import matplotlib.pyplot as plt
from util import *
import tensorflow as tf

import matplotlib
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

## load survey
survey = pd.read_csv('data/survey_processed_1230.csv')
survey['ID'] = survey['ID'].astype(str)
survey.set_index('ID', inplace=True)

## load smart meter data
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
power_df['time'] = pd.to_datetime(power_df['time'])
weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)

#%% valid consumption data flitering
# valid day가 n일 이상 있을 시 dict에 추가
power_dict = dict()
power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48*7))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)
a = []
for i, key in enumerate(power_df.columns):
    if non_nan_idx.sum(axis=1)[i] >=2: # 2주 이상 있음..
        power_dict[key] = []
        # all day
        power_dict[key].append(power_rs[i,non_nan_idx[i],:])

def matching_ids(QID):
    # make label corresponding QID
    label_raw = survey[QID]
    label_raw = label_raw[label_raw != -1]

    data = []
    label = []
    for id in label_raw.index:
        id = str(id)
        if id in power_dict.keys():
            # 24시간 프로파일
            data_tmp = np.mean(power_dict[id][0], axis=0).reshape(1,-1)

            # 48 point를 24 point로 down sampling # FIXME
            data_tmp_down = []
            for i in range(0, 48*7, 2):
                 data_tmp_down.append(np.nanmax(data_tmp[:,i:i+2], axis=1).reshape(-1,1))
            data_tmp_down = np.concatenate(data_tmp_down, axis=1)
            data.append(data_tmp_down)
            label.append(label_raw.loc[id])
    data = np.concatenate(data, axis=0)
    label = np.array(label)
    return data, label

def matching_ids_with_aug(QID):
    # make label corresponding QID
    label_raw = survey[QID]
    label_raw = label_raw[label_raw != -1]

    data = []
    label = []
    for id in label_raw.index:
        id = str(id)
        if id in power_dict.keys():
            # 24시간 프로파일
            non_nan_idx = np.all(~pd.isnull(power_dict[id][0]), axis= 1)
            data_tmp = power_dict[id][0][non_nan_idx,:]
            data_tmp = data_tmp[:2,:] # 2주만 가져옴
            data.append(data_tmp)
            label.append(np.repeat(label_raw.loc[id], 14))
    data = np.concatenate(data, axis=0)
    label = np.ravel(np.array(label))
    return data, label

#%% 데이터 저장
from tqdm import tqdm

data_dict, label_dict, data_ref_dict, label_ref_dict = dict(), dict(), dict(), dict()
for i in tqdm(range(1,16)):
    data_ref, label_ref = matching_ids('Q'+str(i))
    data, label = matching_ids_with_aug('Q' + str(i))

    label_ref_dict['Q' + str(i)] = label_ref
    data_ref_dict['Q' + str(i)] = data_ref

    data_dict['Q'+str(i)] = data
    label_dict['Q'+str(i)] = label

#%% CER 데이터 학습 -- CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K

def baseline_model(lr, binary):

    x_input = Input(shape=(7, 24, 1))
    x = Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1),
           kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
           bias_initializer=initializers.Ones())(x_input)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
           bias_initializer=initializers.Ones())(x)
    x = MaxPool2D()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(32, input_shape=(320,), kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
           bias_initializer=initializers.Ones())(x)

    # Add svm layer
    if binary:
        x_out = Dense(1, input_shape=(32,),
                      activation='linear',use_bias=False, name='svm', kernel_regularizer=regularizers.l2(0.0001))(x)
    else:
        x_out = Dense(label.shape[1], input_shape=(32,),
                      use_bias=False, activation='linear', name='svm', kernel_regularizer=regularizers.l2(0.0001))(x)

    model = Model(x_input, x_out)

    metrics = ['accuracy']

    model.compile(optimizer=Adam(), loss='squared_hinge', metrics=metrics)

    return model

params = {
    'lr': 0.1,
    'epoch': 100
}

result_CNN = []
# for i in tqdm(range(1, 16)):
for i in tqdm([3]):
    data, label_ref = data_ref_dict['Q'+str(i)], label_ref_dict['Q'+str(i)]
    label_ref = label_ref.astype(float)
    binary = False
    label = to_categorical(label_ref.copy(), dtype=int)

    # binary or multi
    if np.all(np.unique(label_ref) == np.array([0, 1])):
        binary = True
        label = label[:, 1]

    y_pred = np.zeros(label.shape)
    y_true = np.zeros(label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    label = label.astype(float)
    label[label == 0] = -1

    for train_index, test_index in skf.split(data, label_ref):

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        """ 
        CNN based feature selection
        """
        # reshape
        X_train = X_train.reshape(-1, 7, 24, 1)
        X_test = X_test.reshape(-1, 7, 24, 1)
        model = baseline_model(params['lr'], binary)
        es = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )
        model.fit(X_train, y_train, epochs=params['epoch'], verbose=1, callbacks=[es], validation_data=(X_test, y_test),
                  batch_size=128)

        if binary:
            y_pred[test_index] = model.predict(X_test).reshape(-1)
        else:
            y_pred[test_index] = model.predict(X_test)
        y_true[test_index] = y_test

    if binary:
        y_pred = np.sign(y_pred)
        result = (y_pred == y_true).mean()
    else:
        result = (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).mean()
    result_CNN.append(result)
    # model.save("models/model_Q_"+str(i)+'.h5')
print(result_CNN)


#%% CER 데이터 학습 -- rf
from sklearn.ensemble import RandomForestClassifier

result_rf = []
for i in tqdm(range(1, 16)):
    data, label = data_ref_dict['Q'+str(i)], label_ref_dict['Q'+str(i)]
    y_pred = np.zeros(label.shape)
    y_true = np.zeros(label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for train_index, test_index in tqdm(skf.split(data, label)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = RandomForestClassifier(n_estimators=100, random_state=0, max_features=5, verbose=True,
                                       n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)
        y_true[test_index] = y_test

    result = (y_pred == y_true).mean()

    result_rf.append(result)

print(result_rf)