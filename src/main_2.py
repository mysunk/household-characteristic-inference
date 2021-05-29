#%%
# ----------------------------------------------------------------------------------------------------------
# Experiment 2. Typical transfer learning
# ----------------------------------------------------------------------------------------------------------

import tensorflow as tf
import matplotlib
import warnings

font = {'size': 16, 'family': "DejaVu"}
matplotlib.rc('font', **font)

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# tmp configure
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.filterwarnings(action='ignore')

# User input
# parser = argparse.ArgumentParser()
# parser.add_argument("--option", default='self', type=str)
#
# args = parser.parse_args()

# %% load dataset
from module.util_main import load_ETRI

ETRI_question_list = {
    'Q1': '# of residents',
    'Q2': '# of appliances',
    'Q3': 'Have children under 10',
    'Q4': 'Have children under 20',
    'Q5': 'Is single',
    'Q6': 'Floor area'
}

# target data
ETRI_data_dict, ETRI_label_dict = load_ETRI(option='target')

# source data
ETRI_data_ul = load_ETRI(option='source')

# load CER dataset
from module.util_main import load_CER

CER_question_list = {
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
CER_data_dict, CER_label_dict = load_CER(option='target')

# source data
CER_data_ul = load_CER(option='source')  # unlabeled

## matching questions
# ETRI - CER
question_pair = {
    'Q1': 'Q13',
    'Q2': 'Q12',
    'Q3': 'Q3',
    'Q4': 'Q3',
    'Q5': 'Q8',
    'Q6': 'Q5',
}

#%% source model 학습

## 학습 parameter
LEARNING_RATE_SOURCE = 1e-4
LR_DECAY_RATE = tf.math.exp(-0.1)
DECAY_TH = 100
BATCH_SIZE = 128
EPOCH_SIZE = 1000

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from module.util_main import CNN
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model_list_source = dict()
history_list_source = dict()
GNT_list_source = dict()

for QUESTION_T in tqdm(ETRI_question_list.keys()):
    QUESTION_S = question_pair[QUESTION_T]

    # QUESTION에 맞는 source 데이터 로드
    data_source = CER_data_dict[QUESTION_S]
    label_source_raw = CER_label_dict[QUESTION_S]

    # 전처리
    label_source = to_categorical(label_source_raw, dtype=int)
    binary = True if label_source.shape[1] == 2 else False
    label_source = label_source.astype(float)
    X_train_source, X_test_source, y_train_source, y_test_source = train_test_split(data_source, label_source, test_size=0.2, random_state=0, stratify=label_source)

    # reshape
    X_train_source = X_train_source.reshape(-1, 7, 24, 1)
    X_test_source = X_test_source.reshape(-1, 7, 24, 1)

    # model 학습
    model_source = CNN(data_source, label_source)

    # callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)

    def scheduler(epoch, lr):
        return lr if epoch < DECAY_TH else lr * LR_DECAY_RATE

    ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
    optimizer = Adam(LEARNING_RATE_SOURCE)

    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    model_source.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    history = model_source.fit(X_train_source, y_train_source, epochs=EPOCH_SIZE, verbose=0,
                        callbacks=[es, ls], validation_data=(X_test_source, y_test_source), batch_size=BATCH_SIZE)

    model_list_source[QUESTION_T] = model_source
    history_list_source[QUESTION_T] = [history]
    GNT_list_source[QUESTION_T] = y_test_source

#%% source model learning curve
from module.util_main import plot_learning, plot_pred_result

##### 학습 결과 plot (only best)
metric = 'loss'
# option = 'accuracy'

plot_learning(ETRI_question_list, history_list_source, GNT_list_source, metric, N_TRIAL = 1)

# %% target model 학습 (self)
from collections import defaultdict
from module.util_main import PredictionCallback

## 학습 parameter
N_TRIAL = 10 # N_TRIAL 개의 모델 학습
EPOCH_SIZE = 100
LEARNING_RATE_TARGET = 1e-4

# 결과 저장
model_list_target_self = defaultdict(list)
history_list_target_self = defaultdict(list)
question_pred_result_self = defaultdict(list)

for QUESTION_T in tqdm(ETRI_question_list.keys()):
    # QUESTION에 맞는 target 데이터 로드
    data_target = ETRI_data_dict[QUESTION_T]
    label_target_raw = ETRI_label_dict[QUESTION_T]

    # 전처리
    label_target = to_categorical(label_target_raw, dtype=int)
    binary = True if label_target.shape[1] == 2 else  False
    label_target = label_target.astype(float)
    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(data_target, label_target,
                                                                                    stratify=label_target)
    # reshape
    X_train_target = X_train_target.reshape(-1, 7, 24, 1)
    X_test_target = X_test_target.reshape(-1, 7, 24, 1)

    for iters in range(N_TRIAL):
        # model 학습
        model_target = CNN(data_target, label_target)

        # callbacks
        es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
        def scheduler(epoch, lr):
            return lr if epoch < DECAY_TH else lr * LR_DECAY_RATE
        ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
        pc = PredictionCallback(X_test_target)

        optimizer = Adam(LEARNING_RATE_TARGET)
        loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
        model_target.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = model_target.fit(X_train_target, y_train_target, epochs=EPOCH_SIZE, verbose=0,
                                   callbacks=[es, ls, pc], validation_data=(X_test_target, y_test_target),
                                   batch_size=BATCH_SIZE)

        model_list_target_self[QUESTION_T].append(model_target)
        history_list_target_self[QUESTION_T].append(history)
        question_pred_result_self[QUESTION_T].append(pc.y_preds)


# %% target model 학습 (transfer)
model_list_target_transfer = defaultdict(list)
question_pred_result_transfer = defaultdict(list)
history_list_target_transfer = defaultdict(list)
GNT_list = dict()

## 학습 parameter
LEARNING_RATE = LEARNING_RATE_SOURCE * 0.1 # source의 0.1배
GLOBAL = True # True시 global fine tuning

for QUESTION_T in tqdm(ETRI_question_list.keys()):
    # QUESTION에 맞는 target 데이터 로드
    data_target = ETRI_data_dict[QUESTION_T]
    label_target_raw = ETRI_label_dict[QUESTION_T]

    # 전처리
    label_target = to_categorical(label_target_raw, dtype=int)
    binary = True if label_target.shape[1] == 2 else False
    label_target = label_target.astype(float)
    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(data_target, label_target,
                                                                                    stratify=label_target)

    # reshape
    X_train_target = X_train_target.reshape(-1, 7, 24, 1)
    X_test_target = X_test_target.reshape(-1, 7, 24, 1)

    for iters in range(N_TRIAL):
        # model 학습
        model_target = CNN(data_target, label_target)

        # transfer : prediction layer 제외하고 transfer
        model_target.layer_1 = model_list_source[QUESTION_T].layer_1
        model_target.layer_2 = model_list_source[QUESTION_T].layer_2
        if not GLOBAL:
            model_target.layer_1.trainable = False
            model_target.layer_2.trainable = False

        # callbacks
        es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
        def scheduler(epoch, lr):
            return lr if epoch < DECAY_TH else lr * LR_DECAY_RATE
        ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
        pc = PredictionCallback(X_test_target)

        optimizer = Adam(LEARNING_RATE)
        loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
        model_target.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = model_target.fit(X_train_target, y_train_target, epochs=EPOCH_SIZE, verbose=0,
                                   callbacks=[es, ls, pc], validation_data=(X_test_target, y_test_target),
                                   batch_size=BATCH_SIZE)

        model_list_target_transfer[QUESTION_T].append(model_target)
        history_list_target_transfer[QUESTION_T].append(history)
        question_pred_result_transfer[QUESTION_T].append(pc.y_preds)

    GNT_list[QUESTION_T] = y_test_target

#%% concatenate history
histories = dict()
question_pred_result = dict()
for QUESTION in tqdm(ETRI_question_list.keys()):
    histories[QUESTION] = history_list_target_self[QUESTION] + history_list_target_transfer[QUESTION]
    question_pred_result[QUESTION] = question_pred_result_self[QUESTION] + question_pred_result_transfer[QUESTION]

#%% self-transfer 결과 분석 및 비교
from module.util_main import plot_pred_result_v2
##### 학습 결과 plot (only best)
metric = 'loss'
# metric = 'accuracy'

plot_learning(ETRI_question_list, history_list_target_self, GNT_list, metric, N_TRIAL)
plot_learning(ETRI_question_list, history_list_target_transfer, GNT_list, metric, N_TRIAL)

#### Prediction 결과 plot
option = 'best'
# option = 'all'

# 따로
plot_pred_result(ETRI_question_list, history_list_target_self, question_pred_result_self, GNT_list, option, metric,  EPOCH_SIZE, N_TRIAL)
plot_pred_result(ETRI_question_list, history_list_target_transfer, question_pred_result_transfer, GNT_list, option, metric,  EPOCH_SIZE, N_TRIAL)

# 같이
plot_pred_result_v2(ETRI_question_list, histories, question_pred_result, GNT_list, 'all', metric,  EPOCH_SIZE, N_TRIAL * 2)