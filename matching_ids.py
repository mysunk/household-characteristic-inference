import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

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
power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)
a = []
for i, key in enumerate(power_df.columns):
    if non_nan_idx.sum(axis=1)[i] >=14:
        power_dict[key] = []
        # all day
        power_dict[key].append(power_rs[i,non_nan_idx[i],:])
        # week day
        power_dict[key].append(power_rs[i, (weekday_flag <5) * non_nan_idx[i], :])
        # week end
        power_dict[key].append(power_rs[i, (weekday_flag >=5) * non_nan_idx[i], :])

#%%
def feature_extraction(profile_allday, profile_weekday, profile_weekend):
    features = []
    '''
        mean features
    '''
    # 1. mean P, daily, all day : c_total
    feature = np.mean(profile_allday, axis=1).mean()
    features.append(feature)

    # 2. mean P, daily, weekday : c_weekday
    feature = np.mean(profile_weekday, axis=1).mean()
    features.append(feature)

    # 3. mean P, daily, weekend : c_weekend
    feature = np.mean(profile_weekend, axis=1).mean()
    features.append(feature)

    # 4. mean P, 6 ~ 22 : c_day
    feature = np.mean(profile_allday[:, 2 * 6:2 * 22], axis=1).mean()
    features.append(feature)

    # 5. mean P, 6 ~ 8.5: c_morning
    feature = np.mean(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1).mean()
    features.append(feature)

    # 6. mean P, 8.5 ~ 12: c_forenoon
    feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1).mean()
    features.append(feature)

    # 7. mean P, 12 ~ 14.5: c_noon
    feature = np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1).mean()
    features.append(feature)

    # 8. mean P, 14.5 ~ 18: c_afternoon
    feature = np.mean(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1).mean()
    features.append(feature)

    # 9. mean P, 18 ~ 24: c_evening
    feature = np.mean(profile_allday[:, 2 * 18:2 * 24], axis=1).mean()
    features.append(feature)

    # 10. mean P, 00 ~ 6: c_night
    feature = np.mean(profile_allday[:, :2 * 6], axis=1).mean()
    features.append(feature)

    '''
        max features
    '''
    # 1. max P, daily, all day : c_total
    feature = np.max(profile_allday, axis=1).mean()
    features.append(feature)

    # 2. max P, daily, weekday : c_weekday
    feature = np.max(profile_weekday, axis=1).mean()
    features.append(feature)

    # 3. max P, daily, weekend : c_weekend
    feature = np.max(profile_weekend, axis=1).mean()
    features.append(feature)

    # 4. max P, 6 ~ 22 : c_day
    feature = np.max(profile_allday[:, 2 * 6:2 * 22], axis=1).mean()
    features.append(feature)

    # 5. max P, 6 ~ 8.5: c_morning
    feature = np.max(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1).mean()
    features.append(feature)

    # 6. max P, 8.5 ~ 12: c_forenoon
    feature = np.max(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1).mean()
    features.append(feature)

    # 7. max P, 12 ~ 14.5: c_noon
    feature = np.max(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1).mean()
    features.append(feature)

    # 8. max P, 14.5 ~ 18: c_afternoon
    feature = np.max(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1).mean()
    features.append(feature)

    # 9. max P, 18 ~ 24: c_evening
    feature = np.max(profile_allday[:, 2 * 18:2 * 24], axis=1).mean()
    features.append(feature)

    # 10. max P, 00 ~ 6: c_night
    feature = np.max(profile_allday[:, :2 * 6], axis=1).mean()
    features.append(feature)
    '''
        min
    '''
    # 1. min P, daily, all day : c_total
    feature = np.min(profile_allday, axis=1).mean()
    features.append(feature)

    # 2. min P, daily, weekday : c_weekday
    feature = np.min(profile_weekday, axis=1).mean()
    features.append(feature)

    # 3. min P, daily, weekend : c_weekend
    feature = np.min(profile_weekend, axis=1).mean()
    features.append(feature)

    # 4. min P, 6 ~ 22 : c_day
    feature = np.min(profile_allday[:, 2 * 6:2 * 22], axis=1).mean()
    features.append(feature)

    # 5. min P, 6 ~ 8.5: c_morning
    feature = np.min(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1).mean()
    features.append(feature)

    # 6. min P, 8.5 ~ 12: c_forenoon
    feature = np.min(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1).mean()
    features.append(feature)

    # 7. min P, 12 ~ 14.5: c_noon
    feature = np.min(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1).mean()
    features.append(feature)

    # 8. min P, 14.5 ~ 18: c_afternoon
    feature = np.min(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1).mean()
    features.append(feature)

    # 9. min P, 18 ~ 24: c_evening
    feature = np.min(profile_allday[:, 2 * 18:2 * 24], axis=1).mean()
    features.append(feature)

    # 10. min P, 00 ~ 6: c_night
    feature = np.min(profile_allday[:, :2 * 6], axis=1).mean()
    features.append(feature)
    '''
        ratios
    '''
    # 1. mean P over max P
    feature = np.mean(profile_allday, axis=1) / np.max(profile_allday, axis=1)
    feature = feature.mean()
    features.append(feature)

    # 2. min P over mean P
    feature = np.min(profile_allday, axis=1) / np.mean(profile_allday, axis=1)
    feature = feature.mean()
    features.append(feature)

    # 3. c_forenoon / c_noon
    feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1) / \
              np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    feature = feature.mean()
    features.append(feature)

    # 4. c_afternoon / c_noon
    feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1) / \
              np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    feature = feature.mean()
    features.append(feature)

    # 5. c_evening / c_noon
    feature = np.mean(profile_allday[:, 2 * 18:2 * 24], axis=1) / \
              np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    feature = feature.mean()
    features.append(feature)

    # 6. c_noon / c_total
    feature = np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1) / \
              np.mean(profile_allday, axis=1)
    feature = feature.mean()
    features.append(feature)

    # 7. c_night / c_day
    feature = np.mean(profile_allday[:, :2 * 6], axis=1) / \
              np.mean(profile_allday[:, 2 * 6:2 * 22], axis=1)
    feature = feature.mean()
    features.append(feature)

    # 8. c_weekday / c_weekend
    feature = np.mean(profile_weekday, axis=1).mean() / \
              np.mean(profile_weekend, axis=1).mean()
    features.append(feature)
    '''
        temporal properties
    '''
    # 1. P > 0.5
    feature = (profile_allday > 0.5).mean(axis=1).mean()
    features.append(feature)

    # 2. P > 0.5
    feature = (profile_weekday > 0.5).mean(axis=1).mean()
    features.append(feature)

    # 3. P > 0.5
    feature = (profile_weekend > 0.5).mean(axis=1).mean()
    features.append(feature)

    # 1. P > 1
    feature = (profile_allday > 1).mean(axis=1).mean()
    features.append(feature)

    # 2. P > 1
    feature = (profile_weekday > 1).mean(axis=1).mean()
    features.append(feature)

    # 3. P > 1
    feature = (profile_weekend > 1).mean(axis=1).mean()
    features.append(feature)

    # 1. P > 2
    feature = (profile_allday > 2).mean(axis=1).mean()
    features.append(feature)

    # 2. P > 2
    feature = (profile_weekday > 2).mean(axis=1).mean()
    features.append(feature)

    # 3. P > 2
    feature = (profile_weekend > 2).mean(axis=1).mean()
    features.append(feature)

    '''
        statistical properties
    '''
    # 1. var P, daily, all day : c_total
    feature = np.var(profile_allday, axis=1).mean()
    features.append(feature)

    # 2. var P, daily, weekday : c_weekday
    feature = np.var(profile_weekday, axis=1).mean()
    features.append(feature)

    # 3. var P, daily, weekend : c_weekend
    feature = np.var(profile_weekend, axis=1).mean()
    features.append(feature)

    # 4. var P, 6 ~ 22 : c_day
    feature = np.var(profile_allday[:, 2 * 6:2 * 22], axis=1).mean()
    features.append(feature)

    # 5. var P, 6 ~ 8.5: c_morning
    feature = np.var(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1).mean()
    features.append(feature)

    # 6. var P, 8.5 ~ 12: c_forenoon
    feature = np.var(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1).mean()
    features.append(feature)

    # 7. var P, 12 ~ 14.5: c_noon
    feature = np.var(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1).mean()
    features.append(feature)

    # 8. var P, 14.5 ~ 18: c_afternoon
    feature = np.var(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1).mean()
    features.append(feature)

    # 9. var P, 18 ~ 24: c_evening
    feature = np.var(profile_allday[:, 2 * 18:2 * 24], axis=1).mean()
    features.append(feature)

    # 10. var P, 00 ~ 6: c_night
    feature = np.var(profile_allday[:, :2 * 6], axis=1).mean()
    features.append(feature)

    return np.array(features).reshape(1,-1)

def matching_ids(QID):
    # make label corresponding QID
    label_raw = survey[QID]
    label_raw = label_raw[label_raw != -1]

    data = []
    label = []
    for id in label_raw.index:
        id = str(id)
        if id in power_dict.keys():
            # data_tmp = feature_extraction(power_dict[id][0], power_dict[id][1], power_dict[id][2])
            # 24시간 프로파일
            data_tmp = np.mean(power_dict[id][0], axis=0).reshape(1,-1)
            data.append(data_tmp)
            label.append(label_raw.loc[id])
    data = np.concatenate(data, axis=0)
    label = np.array(label)
    return data, label

#%% 데이터 저장
data_dict, label_dict = dict(), dict()
for i in range(1,16):
    data, label = matching_ids('Q'+str(i))
    data_dict['Q'+str(i)] = data
    label_dict['Q'+str(i)] = label

#%% 간단한 rf 모델
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
for i in range(1,16):
    data, label = data_dict['Q'+str(i)], label_dict['Q'+str(i)]

    y_pred = np.zeros(label.shape)
    y_true = np.zeros(label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in tqdm(skf.split(data, label)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = RandomForestClassifier(n_estimators=100, random_state=0, max_features=5)
        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)
        y_true[test_index] = y_test

    # acc
    print((y_pred == y_true).mean())

#%% 간단한 svr 모델
from sklearn import svm
for i in range(1,16):
    data, label = data_dict['Q'+str(i)], label_dict['Q'+str(i)]

    y_pred = np.zeros(label.shape)
    y_true = np.zeros(label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in tqdm(skf.split(data, label)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = svm.SVC(decision_function_shape='ovo',gamma='auto',kernel='rbf')
        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)
        y_true[test_index] = y_test

    # acc
    print((y_pred == y_true).mean())