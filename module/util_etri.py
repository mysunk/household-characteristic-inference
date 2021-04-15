import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data_path = 'D:/GITHUB/python_projects/etri-load-preprocessing/'

def find_nan(homes, i, nan_l, nan_h, drop):
    home = homes.iloc[:, i].reset_index(drop=True)
    # zero를 nan으로 바꿈
    zero_index = (home.values == 0)
    home.loc[zero_index] = np.nan

    # nan의 start와 end를 구함
    start_i, end_i = -1, -1
    nan_val = home.isna().values
    nan_indexes = []
    for index, _ in enumerate(nan_val[:-1]):
        if nan_val[index] == False and nan_val[index + 1] == True:
            start_i = index + 1
            continue
        if nan_val[index] == True and nan_val[index + 1] == False:
            end_i = index
            # 처음 시작한 경우
            if start_i == -1:
                nan_indexes.append((0, end_i))
            else:
                nan_indexes.append((start_i, end_i))
            start_i, end_i = -1, -1
            continue
        # 마지막인 경우
        if index == len(nan_val) - 2 and start_i != -1:
            nan_indexes.append((start_i, index + 1))

    # nan이 nan_l <= ~ <= nan_h 일 경우 불러옴
    home = home.values
    # length and peak intensity
    # avg and sigma
    avg = np.nanmean(np.reshape(home, (-1, 24)), axis=0)
    sig = np.nanstd(np.reshape(home, (-1, 24)), axis=0)

    l_p_b = []  # 직전 값과의 corr
    nan_index_selected = []
    for i, nan_index in enumerate(nan_indexes):
        nan_length = nan_index[1] - nan_index[0] + 1
        if (nan_length > nan_l) * (nan_length < nan_h):  # 조건을 만족할때만
            if i == 0:
                continue
            # if nan_index[0] in cand_index:
            #    continue

            # 3 sigma를 넘어갈 때만
            time = (nan_index[0] - 1) % 24
            tmp = home[nan_index[0] - 1] < avg[time] - 3 * sig[time] or home[nan_index[0] - 1] > avg[time] + 3 * sig[
                time]
            if drop:
                if tmp:
                    l_p_b.append((nan_length, home[nan_index[0] - 1]))
                    nan_index_selected.append(nan_index)
            else:
                l_p_b.append((nan_index[0]-1, home[nan_index[0] - 1]))
                nan_index_selected.append(nan_index[0])

    return nan_index_selected, l_p_b

import itertools
import seaborn as sns

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = (cm.astype('float16')/ cm.sum(axis=1)[:, np.newaxis] * 100).astype(int)/100

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.show()

def load_calendar(start_date,end_date):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    datas = []
    data = pd.read_csv(data_path + 'data_raw/calendar/calendar2017.csv', header=None)
    datas.append(data)
    data = pd.read_csv(data_path + 'data_raw/calendar/calendar2018.csv', header=None)
    datas.append(data)
    data = pd.read_csv(data_path + 'data_raw/calendar/calendar2019.csv', header=None)
    datas.append(data)
    calendar = pd.concat(datas, axis=0, join='outer')
    calendar = calendar.reset_index(drop=True)
    calendar.columns = ['year','month','day','dayofweek','wnw']
    calendar['date'] = calendar['year'].astype(str)\
                       +'-'+calendar['month'].astype(str)\
                       +'-'+calendar['day'].astype(str)
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar.index = calendar['date']
    calendar = calendar.loc[start_date:end_date, :]
    calendar = calendar.loc[:, 'wnw']
    # calendar = calendar.astype(bool)
    return calendar

def load_energy(start_date,end_date, idx=None):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_raw = pd.read_csv(data_path + 'data_raw/label_data.csv', low_memory=False, dtype={
        'Time': str,
        'Season': str,
        'Weekday': str,
    })
    data_raw.index = pd.to_datetime(data_raw['Time'])
    data_raw = data_raw.loc[start_date:end_date, :]
    data_raw[(data_raw.values == 0)] = np.nan
    # extra info 데이터와 순서를 맞춤
    people_n = pd.read_excel(data_path + 'data/people_num.xlsx', header=None, index_col=0)
    extra = people_n.iloc[:, 0:2]
    people_n = people_n.iloc[:, 2:]
    people_n[np.isnan(people_n)] = 0
    people_n = people_n.astype(int)
    ## 집 순서와 label 순서 동일하게
    data = data_raw.iloc[:, :3]
    for i, home in enumerate(people_n.index):
        index = np.where(data_raw.columns == home)[0]
        if len(index) == 1:
            date = extra.loc[home, 2]
            try:
                date = date[:7]
            except:
                date = str(date)[:7]
            try:
                date = pd.to_datetime(date)
            except:
                date = pd.to_datetime(date[:4])

            if date > pd.to_datetime(start_date):
                # print(date)
                pass
            else:
                data[home] = data_raw.iloc[:, index[0]]
        else:
            # print(f'index is {index} ===')
            continue
    data = data.loc[start_date:end_date, :].copy()
    data.drop(columns=['Time', 'Season', 'Weekday'], inplace=True)
    if idx is not None:
        sub_idx = [i for i in idx if i in data.columns]
        data = data.loc[:,sub_idx]
    return data

def load_extra_info(idx):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    people_n = pd.read_excel(data_path + 'data/people_num.xlsx', header=None, index_col=0)
    appliance = pd.read_excel(data_path + 'data/appliance.xlsx', index_col=0, header=None)
    working_info = pd.read_excel(data_path + 'data/working_info.xlsx', index_col=0, header=None)

    extra = people_n.iloc[:, 0:2]
    people_n = people_n.iloc[:, 2:]
    people_n[np.isnan(people_n)] = 0
    people_n = people_n.astype(int)
    extra_info = pd.DataFrame(columns=idx)
    for col in idx:
        extra_info[col] = np.concatenate(
            [np.array([extra.loc[col, 1]]), appliance.loc[col, :].values, people_n.loc[col, :].values,
             working_info.loc[col, :].values])
    extra_info[np.isnan(extra_info.values)] = 0
    extra_info = extra_info.astype(int)
    extra_info.index = ['area',
                        'ap_1', 'ap_2', 'ap_3', 'ap_4', 'ap_5', 'ap_6',
                        'ap_7', 'ap_8', 'ap_9','ap_10', 'ap_11',
                        'm_0', 'm_10', 'm_20', 'm_30', 'm_40', 'm_50', 'm_60', 'm_70',
                        'w_0', 'w_10', 'w_20', 'w_30', 'w_40', 'w_50', 'w_60', 'w_70',
                        'income_type','work_type']
    return extra_info


#%% Entropy
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    # print(count)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

#Information Gain
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y,X)


def quantz(X, step = 0.1, bins=100):
    X_quan = np.zeros(X.shape).astype(int)
    for i in range(bins):
        index = (X > step * i) * (X <= step * (i+1))
        X_quan[index] = i+1
    # 10을 넘는 부분
    index = (X > 0.1 * bins)
    X_quan[index] = bins+1
    return X_quan

def jentropy(X, Y):
    probs = []
    for c1 in set(X):
        for c2 in set(Y):
            probs.append(np.mean(np.logical_and(X == c1, Y == c2)))
    return np.sum(-p * np.log(p) for p in probs)