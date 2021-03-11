# ----------------------------------------------------------------------------------------------------------
# Experiment 1. AE based transfer learning
# ----------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import argparse
import warnings
font = {'size': 16, 'family':"DejaVu"}
matplotlib.rc('font', **font)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

warnings.filterwarnings(action='ignore')

# User input
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='ETRI', type=str)

args = parser.parse_args()


#%% load dataset
if args.dataset == 'ETRI':
    from util_main import load_ETRI

    question_list = {
        'Q1': '# of residents',
        'Q2': '# of appliances',
        'Q3': 'Have children under 10',
        'Q4': 'Have children under 20',
        'Q5': 'Is single',
        'Q6': 'Floor area'
    }

    # target data
    data_dict, label_dict = load_ETRI(option = 'target')

    # source data
    data_ul = load_ETRI(option = 'source')
else:
    # load CER dataset
    from util_main import load_CER

    question_list = {
        'Q1': 'Age of building',
        'Q2': 'Retired',
        'Q3': 'Have children',
        'Q4': 'Age of building',
        'Q5': 'Floor area',
        'Q6': 'Buld proportion',
        'Q7': 'House type',
        'Q8': 'Is single',
        'Q9': 'Is family',
        'Q10': 'Cooking facility type',
        'Q11': '# of bedrooms',
        'Q12': '# of appliances',
        'Q13': '# of residents',
        'Q14': 'Social class',
        'Q15': 'House occupancy'
    }

    # target data
    data_dict, label_dict = load_CER(option = 'target')

    # source data
    data_ul = load_CER(option ='source') # unlabeled


#%%
##### AE 모델

## 학습 parameter
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPOCH_SIZE = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128

## 학습
from util_main import Autoencoder
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

train_data = data_ul.reshape(-1, 7, 24, 1)

base_model = Autoencoder()
optimizer = SGD(LEARNING_RATE, momentum=MOMENTUM)
base_model.compile(optimizer = optimizer, loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
history = base_model.fit(train_data, train_data, epochs=EPOCH_SIZE, verbose=1,
          batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks = [es])

base_model_2 = Autoencoder()
optimizer = SGD(LEARNING_RATE, momentum=MOMENTUM)
base_model_2.compile(optimizer = optimizer, loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
history_2 = base_model_2.fit(train_data, train_data, epochs=1, verbose=1,
          batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks = [es])

# sample prediction plot
import matplotlib.pyplot as plt
pred = base_model.predict(train_data[0:1])
pred_2 = base_model_2.predict(train_data[0:1])
plt.plot(train_data[0:1].reshape(-1), label='original')
plt.plot(pred_2.reshape(-1), label = '10 EPOCHs')
plt.plot(pred.reshape(-1),':', label ='{} EPOCHs'.format(EPOCH_SIZE))
plt.xlabel('Hour')
plt.ylabel('Energy consumption [kW]')
plt.legend()
plt.show()

# plot learning curve
from util_main import plot_history
plot_history([('EPOCH: {}'.format(EPOCH_SIZE), history)])

#%%
##### label 데이터 학습
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from util_main import CNN, PredictionCallback
from collections import defaultdict
from tqdm import tqdm

## 학습 parameter
N_TRIAL = 1 # N_TRIAL 개의 모델 학습
LEARNING_RATE = 1e-4
LR_DECAY_RATE = tf.math.exp(-0.1)
DECAY_TH = 40
BATCH_SIZE = 128
EPOCH_SIZE = 10
TRAINED = True # True: using pre-trained model

# 결과 저장
question_pred_result = dict(list)
histories = defaultdict(list)
GNT_list = dict()

## 모든 question에 관해 학습
for QUESTION in tqdm(question_list.keys()):
    # QUESTION에 맞는 데이터 로드
    data = data_dict[QUESTION].copy()
    label_raw = label_dict[QUESTION].copy()

    # 전처리
    label = to_categorical(label_raw, dtype=int)
    binary = True if label.shape[1] == 2 else False
    label = label.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
    # reshape
    X_train = X_train.reshape(-1, 7, 24, 1)
    X_test = X_test.reshape(-1, 7, 24, 1)

    for iters in range(N_TRIAL):

        # model load
        model = CNN(X_train, y_train)

        # replace with pre-trained model
        if TRAINED:
            model.layer_1 = base_model

        # callbacks
        pc = PredictionCallback(X_test)
        es = EarlyStopping(monitor='val_loss',patience=10,verbose=0,mode='min',restore_best_weights=True)
        def scheduler(epoch, lr):
            return lr if epoch < DECAY_TH else lr * LR_DECAY_RATE
        ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
        optimizer = Adam(LEARNING_RATE)

        loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # 모델 학습
        history = model.fit(X_train, y_train, epochs=EPOCH_SIZE, verbose=0,
                        callbacks=[es, ls, pc], validation_data=(X_test, y_test),batch_size=BATCH_SIZE)

        # 결과 저장
        histories[QUESTION].append(history)
        question_pred_result.append(pc.y_preds)

    GNT_list[QUESTION] = y_test

#%% 결과 분석
from util_main import plot_learning, plot_pred_result

##### 학습 결과 plot (only best)
metric = 'loss'
# option = 'accuracy'

plot_learning(question_list, histories, GNT_list, metric, N_TRIAL)

#### Prediction 결과 plot

# option = 'best'
option = 'all'
metric = 'loss'
# metric = 'accuracy'

plot_pred_result(question_list, histories, question_pred_result, GNT_list, option, metric,  EPOCH_SIZE, N_TRIAL)

# ##### pca coefficient 분석
# plt.plot(pca.components_.T)
# plt.xlabel('coefficient index')
# plt.title('PCA coefficient')
# plt.legend(['1st component','2nd component'])
# plt.show()