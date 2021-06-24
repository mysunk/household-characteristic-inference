'''
@ author: Myungsun Kim
@ Implementation date: June 24, 2021

@ paper:
Yan, Siqing, et al. 
"Time–Frequency Feature Combination Based Household Characteristic Identification Approach Using Smart Meter Data." 
IEEE Transactions on Industry Applications 56.3 (2020): 2251-2262.
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

print(CER.shape)
print(CER_label.shape)

print(CER.shape)

#### CER 거르기
# valid_household_idx = (pd.isnull(CER).sum(axis=0) <= 15)
# CER, CER_label = CER.loc[:,valid_household_idx], CER_label.loc[valid_household_idx,:]

# %% feature extraction functions
from pywt import wavedec
def feature_extract_frequencydomain(data):
    '''
    instance가 아닌 전체 데이터셋에 대해서
    '''
    features = []

    ca3, cd3, cd2, cd1 = wavedec(data, 'db3', level = 3)
    ca2, _, _ = wavedec(data, 'db3', level = 2)
    ca1, _ = wavedec(data, 'db3', level = 1)

    def statistical(features_tmp):
        mean_= np.mean(features_tmp, axis=1).reshape(-1, 1)
        max_= np.max(features_tmp, axis=1).reshape(-1, 1)
        min_= np.min(features_tmp, axis=1).reshape(-1, 1)
        var_= np.var(features_tmp, axis=1).reshape(-1, 1)
        return np.concatenate([mean_, max_, min_, var_], axis=1)

    # 1. CA1
    features = np.concatenate([statistical(ca1), statistical(ca2),statistical(ca3),\
        statistical(cd1),statistical(cd2),statistical(cd3)], axis=1)
    return features


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


def feature_extraction_timedomain(data, label):
    flag = data.index[::48].weekday < 5
    flag = np.repeat(flag, 48)
    data_rs_all, home_idx_all = transform(data, 24 * 2)
    data_rs_week, home_idx_week = transform(data.iloc[flag,:], 24 * 2)
    data_rs_weekend, home_idx_weekend = transform(data.iloc[~flag,:], 24 * 2)
    list_1 = list(set(list(home_idx_all)).intersection(list(home_idx_week)))
    common_idx = np.array(list(set(list_1).intersection(list(home_idx_weekend))))

    def feature_extract_instance(profile_allday, profile_weekday, profile_weekend, option = 0):
        '''
        option: closed or opened
        '''
        features = []

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
        feature = np.mean(profile_allday[:, 2 * 6:2 * 22 + option], axis=1).mean()
        features.append(feature)

        # 5. mean P, 6 ~ 8.5: c_morning
        feature = np.mean(profile_allday[:, 2 * 6:2 * 8 + 1+ option], axis=1).mean()
        features.append(feature)

        # 6. mean P, 8.5 ~ 12: c_forenoon
        feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12+ option], axis=1).mean()
        features.append(feature)

        # 7. mean P, 12 ~ 14.5: c_noon
        feature = np.mean(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1).mean()
        features.append(feature)

        # 8. mean P, 14.5 ~ 18: c_afternoon
        feature = np.mean(profile_allday[:, 2 * 14 + 1:2 * 18+ option], axis=1).mean()
        features.append(feature)

        # 9. mean P, 18 ~ 24: c_evening
        feature = np.mean(profile_allday[:, 2 * 18:], axis=1).mean()
        features.append(feature)

        # 10. mean P, 00 ~ 6: c_night
        feature = np.mean(profile_allday[:, :2 * 6+ option], axis=1).mean()
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

        # 4. max P, 6 ~ 8.5: c_morning
        feature = np.max(profile_allday[:, 2 * 6:2 * 8 + 1+ option], axis=1).mean()
        features.append(feature)

        # 5. max P, 8.5 ~ 12: c_forenoon
        feature = np.max(profile_allday[:, 2 * 8 + 1:2 * 12+ option], axis=1).mean()
        features.append(feature)

        # 6. max P, 12 ~ 14.5: c_noon
        feature = np.max(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1).mean()
        features.append(feature)

        # 7. max P, 14.5 ~ 18: c_afternoon
        feature = np.max(profile_allday[:, 2 * 14 + 1:2 * 18+ option], axis=1).mean()
        features.append(feature)

        # 8. max P, 18 ~ 24: c_evening
        feature = np.max(profile_allday[:, 2 * 18:], axis=1).mean()
        features.append(feature)

        # 9. max P, 00 ~ 6: c_night
        feature = np.max(profile_allday[:, :2 * 6+ option], axis=1).mean()
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

        # 4. min P, 6 ~ 8.5: c_morning
        feature = np.min(profile_allday[:, 2 * 6:2 * 8 + 1+ option], axis=1).mean()
        features.append(feature)

        # 5. min P, 8.5 ~ 12: c_forenoon
        feature = np.min(profile_allday[:, 2 * 8 + 1:2 * 12+ option], axis=1).mean()
        features.append(feature)

        # 6. min P, 12 ~ 14.5: c_noon
        feature = np.min(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1).mean()
        features.append(feature)

        # 7. min P, 14.5 ~ 18: c_afternoon
        feature = np.min(profile_allday[:, 2 * 14 + 1:2 * 18+ option], axis=1).mean()
        features.append(feature)

        # 8. min P, 18 ~ 24: c_evening
        feature = np.min(profile_allday[:, 2 * 18:], axis=1).mean()
        features.append(feature)

        # 9. min P, 00 ~ 6: c_night
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
        feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12+ option], axis=1) / \
                np.mean(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1)
        feature = feature.mean()
        features.append(feature)

        # 4. c_afternoon / c_noon
        feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12+ option], axis=1) / \
                np.mean(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1)
        feature = feature.mean()
        features.append(feature)

        # 5. c_evening / c_noon
        feature = np.mean(profile_allday[:, 2 * 18:], axis=1) / \
                np.mean(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1)
        feature = feature.mean()
        features.append(feature)

        # 6. c_noon / c_total
        feature = np.mean(profile_allday[:, 2 * 12:2 * 14 + 1+ option], axis=1) / \
                np.mean(profile_allday, axis=1)
        feature = feature.mean()
        features.append(feature)

        # 7. c_night / c_day
        feature = np.mean(profile_allday[:, :2 * 6+ option], axis=1) / \
                np.mean(profile_allday[:, 2 * 6:2 * 22+ option], axis=1)
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

        # 4. var P, 6 ~ 8.5: c_morning
        feature = np.var(profile_allday[:, 2 * 6:2 * 8 + 1+ option], axis=1).mean()
        features.append(feature)

        # 5. var P, 8.5 ~ 12: c_forenoon
        feature = np.var(profile_allday[:, 2 * 8 + 1:2 * 12+ option], axis=1).mean()
        features.append(feature)

        # 6. var P, 12 ~ 14.5: c_noon
        feature = np.var(profile_allday[:, 2 * 12:2 * 14 + 1]+ option, axis=1).mean()
        features.append(feature)

        # 7. var P, 14.5 ~ 18: c_afternoon
        feature = np.var(profile_allday[:, 2 * 14 + 1:2 * 18+ option], axis=1).mean()
        features.append(feature)

        # 8. var P, 18 ~ 24: c_evening
        feature = np.var(profile_allday[:, 2 * 18:], axis=1).mean()
        features.append(feature)

        # 9. var P, 00 ~ 6: c_night
        feature = np.var(profile_allday[:, :2 * 6+ option], axis=1).mean()
        features.append(feature)

        return np.array(features).reshape(1,-1)

    profile_list = []
    profile_raw = []
    for idx in common_idx:
        profile_allday = data_rs_all[home_idx_all == idx]
        profile_weekday = data_rs_week[home_idx_week == idx]
        profile_weekend = data_rs_weekend[home_idx_weekend == idx]
        profile = feature_extract_instance(profile_allday, profile_weekday, profile_weekend)
        profile_list.append(profile)
        profile_raw.append(profile_allday.mean(axis=0).reshape(1, -1))
    data_transformed = np.concatenate(profile_list, axis=0)
    profile_raw = np.concatenate(profile_raw, axis=0)

    label = label[common_idx]
    return data_transformed, profile_raw, label


def feature_extraction(data, label):
    t_domain_features, data_raw, data_label = feature_extraction_timedomain(data, label)
    f_domain_features = feature_extract_frequencydomain(data_raw)
    concat_features = np.concatenate([t_domain_features, f_domain_features], axis=1)
    return concat_features, data_raw, data_label


# %% feature extraction
CER_features, CER_profile, CER_label_t = feature_extraction(CER, CER_label.loc[:,'Q13'].values)

# %% feature selection
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split

def evaluate(y_true, y_pred):
    '''
    return acc, auc, f1 score
    '''
    acc_ = (np.argmax(y_pred, axis=1) == y_true).mean()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:,0], pos_label=0)
    auc_ = metrics.auc(fpr, tpr)
    f1_score_ = metrics.f1_score(y_true, np.argmax(y_pred, axis=1))
    print('accuracy: {:.3f}, auc: {:.3f}, f1 score: {:.3f}'.format(acc_, auc_, f1_score_))

    return acc_, auc_, f1_score_

X = CER_features
y = CER_label_t.copy()
y[y<=2] = 0
y[y>2] = 1

result_df = pd.DataFrame(columns = ['AC','AUC','F1'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.98, random_state = 0)

'''
rf based feature selection
'''
rf = RandomForestClassifier(n_estimators=191, max_features = 6, n_jobs=-1)
rf.fit(X_train, y_train)
valid_feature_idx = rf.feature_importances_ > 0
# result = permutation_importance(rf, X_test, y_test, n_repeats=10,
#                             random_state=42, n_jobs=6)
# valid_feature_idx = result['importances_mean'] > 0
# valid_feature_idx[:] = True

'''
SVM classifier
'''
svc = SVC(kernel = 'rbf', C = 59, gamma = 0.0001, probability=True)
svc.fit(X_train[:,valid_feature_idx], y_train)
y_pred = svc.predict_proba(X_test[:,valid_feature_idx])

'''
evaluate with acc, auc, f1 score
'''
result_df.loc[0] = evaluate(y_test, y_pred)
