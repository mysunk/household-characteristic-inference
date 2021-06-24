import matplotlib
import matplotlib.pyplot as plt
from module.util import *
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

# 데이터 로드 및 전처리
power_df = pd.read_csv('data/power_comb_SME_included.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*100,:] # 4주
power_df['time'] = pd.to_datetime(power_df['time'])
weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)

# SME & residential 구분 code
code = pd.read_excel('data/doc/SME and Residential allocations.xlsx', engine='openpyxl')
residential = code.loc[code['Code'] == 1,'ID'].values
SME = code.loc[code['Code'] == 2,'ID'].values

# split power data
power_residential = power_df[residential.astype(str)]
power_sme = power_df[SME.astype(str)]

# reshape
power_residential_rs = np.reshape(power_residential.values.T, (power_residential.shape[1], -1, 48))
power_residential_rs = np.nanmean(power_residential_rs, axis=1)
power_sme_rs = np.reshape(power_sme.values.T, (power_sme.shape[1], -1, 48))
power_sme_rs = np.nanmean(power_sme_rs, axis=1)

#%% mean값 plot
data_re_list, data_sme_list = [], []
for i in range(100):
    data_re = power_residential.iloc[:,i].values.reshape(-1,48)
    data_sme = power_sme.iloc[:,i].values.reshape(-1,48)
    data_re = np.nanmean(data_re, axis=0)
    data_sme = np.nanmean(data_sme, axis=0)
    data_re_list.append(data_re)
    data_sme_list.append(data_sme)

data_re = np.nanmean(data_re_list, axis=0)
data_sme = np.nanmean(data_sme_list, axis=0)

plt.figure(figsize=(6,4))
plt.plot(data_re, label='Residential')
plt.plot(data_sme, label='SME')
plt.legend()
plt.xlabel('Time [hour]')
plt.title('Average daily profiles of 100 buildings')
plt.ylabel('Consumption [kW]')
plt.xticks(range(48)[::6], range(24)[::3])
plt.show()

#%% classification 0 :: imbalanced 그냥 두고..
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
R_train, R_test = train_test_split(power_residential_rs, test_size=0.3)
S_train, S_test = train_test_split(power_sme_rs, test_size=0.3)

X_train = np.concatenate([R_train,S_train], axis=0)
X_test = np.concatenate([R_test,S_test], axis=0)
y_train = np.concatenate([R_train.shape[0] * [1],S_train.shape[0] * [2]], axis=0)
y_test = np.concatenate([R_test.shape[0] * [1],S_test.shape[0] * [2]], axis=0)

## 학습
# nan to 0
X_train[pd.isnull(X_train)] = 0
X_test[pd.isnull(X_test)] = 0

# shuffle
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=0)

# fit
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
result = (y_pred == y_test).mean()

# pca
pca = PCA(n_components=6)
X_train_tr = pca.fit_transform(X_train)
X_test_tr = pca.transform(X_test)

clf = SVC(gamma='auto')
clf.fit(X_train_tr, y_train)
y_pred = clf.predict(X_test_tr)

result_pca = (y_pred == y_test).mean()

#%% classification 1
# N_TR, N_TE = 385, 100

result = []
result_pca = []
for i in range(100):
    re_train_idx = np.random.randint(0, power_residential.shape[1], 300)
    re_test_idx = re_train_idx[np.random.randint(0, 300, 60)]

    sme_train_idx = np.random.randint(0, power_sme.shape[1], 200)
    sme_test_idx = sme_train_idx[np.random.randint(0, 200, 40)]

    X_train = np.concatenate([power_residential_rs[re_train_idx],power_sme_rs[sme_train_idx]], axis=0)
    X_test = np.concatenate([power_residential_rs[re_test_idx],power_sme_rs[sme_test_idx]], axis=0)

    y_train = np.concatenate([power_residential_rs[re_train_idx].shape[0] * [1],power_sme_rs[sme_train_idx].shape[0] * [2]], axis=0)
    y_test = np.concatenate([power_residential_rs[re_test_idx].shape[0] * [1],power_sme_rs[sme_test_idx].shape[0] * [2]], axis=0)

    print(pd.isnull(X_train).sum())

    ## 학습
    # nan to 0
    X_train[pd.isnull(X_train)] = 0
    X_test[pd.isnull(X_test)] = 0

    # shuffle
    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # fit
    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    result.append((y_pred == y_test).mean())

    # pca
    pca = PCA(n_components=6)
    X_train_tr = pca.fit_transform(X_train)
    X_test_tr = pca.transform(X_test)

    clf = SVC(gamma='auto')
    clf.fit(X_train_tr, y_train)
    y_pred = clf.predict(X_test_tr)

    result_pca.append((y_pred == y_test).mean())

#result
import seaborn as sns
plt.title('Binary classification accuracy')
sns.boxplot(data = [result, result_pca])
plt.ylabel('Accuracy [%]')
plt.xticks([0, 1],['SVM', 'PCA+SVM'])
plt.show()

#%% classification 2
from tqdm import tqdm
result_2 = []
for i in tqdm(range(10)):
    re_select = np.random.randint(0, power_residential.shape[1], 485)

    power_residential_rs = np.reshape(power_residential.values.T, (power_residential.shape[1], -1, 48))
    power_residential_rs = power_residential_rs[re_select]

    power_residential_rs = np.reshape(power_residential_rs, (-1, 48))
    # 하루중 한 포인트라도 nan이 있으면 제거
    nan_idx = np.any(pd.isnull(power_residential_rs), axis=1)
    power_residential_rs = power_residential_rs[~nan_idx,:]

    power_sme_rs = np.reshape(power_sme.values.T, (power_sme.shape[1], -1,48))
    power_sme_rs = np.reshape(power_sme_rs, (-1, 48))
    # 하루중 한 포인트라도 nan이 있으면 제거
    nan_idx = np.any(pd.isnull(power_sme_rs), axis=1)
    power_sme_rs = power_sme_rs[~nan_idx,:]

    from sklearn.model_selection import train_test_split

    X_train_re, X_test_re = train_test_split(power_residential_rs, test_size=0.2, random_state=42)
    X_train_sme, X_test_sme = train_test_split(power_sme_rs, test_size=0.2, random_state=42)

    X_train = np.concatenate([X_train_re, X_train_sme], axis=0)
    X_test = np.concatenate([X_test_re, X_test_sme], axis=0)

    y_train = np.concatenate([[1] * X_train_re.shape[0], [2] * X_train_sme.shape[0]], axis=0)
    y_test = np.concatenate([[1] * X_test_re.shape[0], [2] * X_test_sme.shape[0]], axis=0)

    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    result_2.append((y_pred == y_test).mean())

import seaborn as sns
plt.title('Binary classification accuracy')
sns.boxplot(data = [result[:10] ,result_2])
plt.ylabel('Accuracy [%]')
plt.xticks([0, 1],['Average', 'Augmented'])
plt.show()

#%% SPARSE ~ 논문 그대로 재현

def transform(raw_data, num_select, label_const = 1):
    raw_data_rs = np.reshape(raw_data.values.T, (raw_data.shape[1], -1, 48))
    idx_select = np.random.randint(0, raw_data.shape[1], num_select)
    raw_data_rs = raw_data_rs[idx_select,:,:]

    data = []
    label = []
    for i in range(raw_data_rs.shape[0]):
        for j in range(raw_data_rs.shape[1]):
            if pd.isnull(raw_data_rs[i,j,:]).sum() == 0:
                data.append(raw_data_rs[i,j,:])
                label.append(label_const)
    data = np.array(data)
    label = np.array(label)
    return data, label

data_re, label_re = transform(power_residential, 300, 1)
data_sme, label_sme = transform(power_sme, 200, 2)
# print(data_re.shape[0] + data_sme.shape[0])

X, y = np.append(data_re, data_sme, axis=0), np.append(label_re, label_sme)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print((y_pred == y_test).mean())

# 틀린거 plot
wrong_idx = np.where(y_pred != y_test)[0]
plt.figure(figsize=(6,3))
idx = 5
plt.plot(X_test[wrong_idx[idx],:])
if y_test[wrong_idx[idx]] == 1:
    plt.legend(['Residential'])
if y_test[wrong_idx[idx]] == 2:
    plt.legend(['SME'])
plt.ylabel('Electricity \n consumption [kW]')
plt.xlabel('Time')
plt.show()

from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.title('Confusion matrix')
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')
plt.xticks([0.5, 1.5], ['Residential', 'SME'])
plt.yticks([0.5, 1.5], ['Residential', 'SME'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()