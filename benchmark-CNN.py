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
# survey = pd.read_csv('data/survey_processed_1230.csv')
survey = pd.read_csv('data/survey_processed_0216.csv')

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

# result dict
CNN_result_dict = dict()
CNN_param_dict = dict()
#%% 학습
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from functools import partial
from util import train

tuning_algo = tpe.suggest
space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(0.3)),
    'epoch': 1000,
    'batch_size': hp.quniform('batch_size', 32, 32*20, 32),
    'lambda': hp.loguniform('lambda', np.log(1e-5), np.log(1)),
    'epsilon': hp.loguniform('epsilon', np.log(1e-5), np.log(1)),
}

for question_number in [13, 12, 3, 8, 5]:
    print(f'===={question_number}====')

    # load data corresponding to questtion
    data, label_ref = data_ref_dict['Q' + str(question_number)], label_ref_dict['Q' + str(question_number)]

    # bayes opt
    bayes_trials = Trials()
    objective = partial(train, data = data, label_ref = label_ref, i = question_number, random_state = 1)
    result = fmin(fn=objective, space=space, algo=tuning_algo, max_evals=50, trials=bayes_trials)

    # save the result
    trials = sorted(bayes_trials.results, key=lambda k: k['loss'])
    if question_number in CNN_result_dict.keys():
        if CNN_result_dict[question_number] < trials[0]['loss']:
            CNN_result_dict[question_number] = trials[0]['loss']
            CNN_param_dict[question_number] = trials[0]['params']
            # search한 parameter로 학습
            train(params = CNN_param_dict[question_number],data = data, label_ref = label_ref, i = question_number, save_model=True)
    del objective, result, trials, bayes_trials

#%% CER 데이터 학습 -- rf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

result_rf = []
for i in tqdm(range(1, 16)):
    data, label = data_ref_dict['Q'+str(i)], label_ref_dict['Q'+str(i)]

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)


    model = RandomForestClassifier(n_estimators=100, random_state=0, max_features=5, verbose=True,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = (y_pred == y_test).mean()

    result_rf.append(result)

print(result_rf)
result_rf[7]