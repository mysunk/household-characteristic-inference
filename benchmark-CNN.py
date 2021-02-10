import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import *
import tensorflow as tf
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
data_dict, label_dict, data_ref_dict, label_ref_dict = dict(), dict(), dict(), dict()
for i in range(1,16):
    data_ref, label_ref = matching_ids('Q'+str(i))
    data, label = matching_ids_with_aug('Q' + str(i))

    label_ref_dict['Q' + str(i)] = label_ref
    data_ref_dict['Q' + str(i)] = data_ref

    data_dict['Q'+str(i)] = data
    label_dict['Q'+str(i)] = label

#%% CER 데이터 학습
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

result = []
for i in tqdm([13, 12, 8, 5]):
# for i in tqdm([8]):
    data, label_ref = data_ref_dict['Q'+str(i)], label_ref_dict['Q'+str(i)]
    binary = False
    label = to_categorical(label_ref.copy(), dtype=int)
    if np.all(np.unique(label_ref) == np.array([0,1])):
        binary = True
        label = label[:, 1]

    y_pred = np.zeros(label.shape)
    y_true = np.zeros(label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in skf.split(data, label_ref):

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        """ 
        CNN based feature selection
        """
        # reshape
        X_train = X_train.reshape(-1,7, 24, 1)
        X_test = X_test.reshape(-1, 7, 24, 1)

        #create model
        model = Sequential()
        #add model layers
        model.add(Conv2D(8, kernel_size=(2,3), activation='relu', input_shape=(7,24,1)))
        model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
        model.add(MaxPool2D())
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(32, input_shape = (320,), activation='relu'))
        if binary:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(label.shape[1], activation='softmax'))

        optimizer = SGD(0.01)
        es = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='auto',
            restore_best_weights=True
        )
        if binary:
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['binary_crossentropy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['categorical_crossentropy'])
        model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_data=(X_test, y_test))
        if binary:
            y_pred[test_index] = model.predict(X_test).reshape(-1)
        else:
            y_pred[test_index] = model.predict(X_test)
        y_true[test_index] = y_test

    # acc
    # print((y_pred == y_true).mean())
    if binary:
        result.append(((y_pred > 0.5) == y_true).mean())
    else:
        result.append((np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).mean())
    print(result)

    model.save("model_Q_"+str(i)+'.h5')

# np.save('../results/CNN_CS.npy', np.array(result))


#%% ETRI 데이터 학습
