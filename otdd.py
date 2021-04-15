# %%
# load dataset

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../CER-information-inference')
from module.util_main import *


def dim_reduct(data, window = 24 * 7):
    '''
    data를 window 단위로 nan이 있는 주를 제외하함
    '''
    cut = (data.shape[0] // window) * window
    data = data[:cut, :]

    data_3d = data.reshape(-1, window, data.shape[1])  # day, hour, home
    data_3d = data_3d.transpose(2, 0, 1) # home, day, hour
    # nan_day = np.any(pd.isnull(data_3d), axis=1)
    # data_3d = data_3d[:,~nan_day, :]  # nan이 없는 집 삭제
    data_list = []
    for i in range(data_3d.shape[0]):
        data_2d = data_3d[i,:,:].copy()
        nan_idx = np.any(pd.isnull(data_2d), axis=1)
        data_2d = data_2d[~nan_idx,:]
        if data_2d.size == 0:
            continue
        data_2d = np.mean(data_2d, axis=0)
        data_list.append(data_2d)
    data = np.array(data_list)
    return data

start_date = pd.to_datetime('2018-07-14 00:00:00')
end_date = pd.to_datetime('2018-08-14 23:00:00')

#%% -- deprecated
## PV
path = 'D:/ISP/5. 받은 자료/승욱오빠/Data/Data_7.csv'
PV = pd.read_csv(path, encoding='CP949')
data = PV.loc[:,'1':'24'].values
data = data.reshape(-1, 24*7)
data_dict['PV'] = data / np.max(data, axis=1).reshape(-1,1)
#%% dataset을 어떻게 선택할지?


#%% 같은 지역끼리

from os import listdir
data_dict = dict()
for name in ['서울','대전','광주','인천','나주']:
    path_src = f'D:/ISP/8. 과제/2020 ETRI/data/SG_data_{name}_비식별화/'
    dirs = listdir(path_src)
    for i, dir in enumerate(dirs):
        if 'csv' not in dir:
            continue
        path = path_src + dir
        data_source = load_energy(path, start_date, end_date)
        data_source = data_source.values
        window = 24 * 7
        data_2d = dim_reduct(data_source, window)
        if data_2d.size == 0:
            continue
        data_dict[name + '_'+str(i)] = data_2d
    print('Load data done')

#%% CER
# 데이터 로드 및 전처리
start_date = pd.to_datetime('2009-07-14 00:00:00')
end_date = pd.to_datetime('2009-08-14 23:00:00')

src_path = 'D:/GITHUB/python_projects/CER-information-inference/'
power_df_raw = pd.read_csv(src_path + 'data/power_comb_SME_included.csv')

# 0 to NaN
power_df_raw[power_df_raw==0] = np.nan
power_df_raw['time'] = pd.to_datetime(power_df_raw['time'])
power_df_raw.index = power_df_raw['time']
power_df = power_df_raw.loc[start_date : end_date,:]
# power_df = power_df_raw.iloc[:2*24*28*4,:] # 한달씩 밀어가며
# weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)

# SME & residential 구분 code
code = pd.read_excel(src_path + 'data/doc/SME and Residential allocations.xlsx', engine='openpyxl')

# SME & residential 구분 code
code = pd.read_excel(src_path + 'data/doc/SME and Residential allocations.xlsx', engine='openpyxl')
residential = code.loc[code['Code'] == 1,'ID'].values
SME = code.loc[code['Code'] == 2,'ID'].values

# split power data
power_residential = power_df[residential.astype(str)]
power_sme = power_df[SME.astype(str)]

# cut
window = 48 * 7
cut = (power_residential.shape[0] // window) * window
power_residential = power_residential.iloc[:cut, :]
cut = (power_sme.shape[0] // window) * window
power_sme = power_sme.iloc[:cut, :]

# reshape
power_residential_rs = dim_reduct(power_residential.values, 48*7)
power_sme_rs = dim_reduct(power_sme.values, 48*7)

# downsampling
from module.util_main import downsampling
power_residential_rs = downsampling(power_residential_rs)
power_sme_rs = downsampling(power_sme_rs)

data_dict['CER_R'] = power_residential_rs
# data_dict['CER_SME_'+str(i)] = power_sme_rs

print('Done load data')
del power_df_raw

#%% OT distance for all (24*7)
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

# area = ['서울', '인천', '광주', '나주', '대전', 'CER_R','CER_SME', 'PV']
area = list(data_dict.keys())

abnormal_idx = [30, 31, 34, 37, 40]
for index in sorted(abnormal_idx, reverse=True):
    del area[index]
LEN_AREA = len(area)
result_ot = np.zeros((LEN_AREA,LEN_AREA))
for i in range(LEN_AREA):
    for j in range(LEN_AREA):
        xs = data_dict[area[i]][:100,:]
        xt = data_dict[area[j]][:100,:]

        n = xs.shape[0]
        m = xt.shape[0]
        a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform for each samples

        # loss matrix
        M = ot.dist(xs, xt)
        # M /= M.max()

        print(ot.emd2(a, b, M, numItermax = 1e+8))
        print(f'n is {n} and m is {m}')
        result_ot[i, j] = ot.emd2(a, b, M)

        # if result_ot[i, j] > 1000:
        #     abnormal_idx.append((i, j))

#%% heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'
sns.heatmap(result_ot[:LEN_AREA-1, :LEN_AREA-1])
plt.xticks(np.linspace(0.5,LEN_AREA - 0.5,LEN_AREA), [''] * LEN_AREA)
plt.yticks(np.linspace(0.5,LEN_AREA - 0.5,LEN_AREA), [''] * LEN_AREA)
# plt.title(h)
plt.show()

#%% box plot
import seaborn as sns

# 0을 빼야함..
result_ot_exclude_0 = []
for i in range(LEN_AREA-1):
    result_tmp = []
    for j in range(LEN_AREA-1):
        if i == j:
            continue
        else:
            result_tmp.append(result_ot[i,j])
    result_ot_exclude_0.append(result_tmp)

plt.figure(figsize=(8,4))
sns.boxplot(data = result_ot_exclude_0)
plt.plot(result_ot[-1,:-1], color='r', label='1 month avg')
# plt.plot(result_ot[46,:40], color='b', label='4 month avg')
plt.xticks(range(40)[::3], range(40)[::3])
plt.xlabel('index of dataset')
plt.ylabel('Optimal distance')
plt.title('optimal distances b.t.w area')
plt.legend()
plt.show()

#%% OT distance w.r.t hour
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

# area = ['서울', '인천', '광주', '나주', '대전', 'CER_R','CER_SME', 'PV']
area = list(data_dict.keys())

abnormal_idx = [30, 31, 34, 37, 40]
for index in sorted(abnormal_idx, reverse=True):
    del area[index]
LEN_AREA = len(area)
result_ot = np.zeros((LEN_AREA,LEN_AREA, 24))
for h in range(24):
    for i in range(LEN_AREA):
        for j in range(LEN_AREA):
            xs = data_dict[area[i]][:100,h:h+1]
            xt = data_dict[area[j]][:100,h:h+1]

            n = xs.shape[0]
            m = xt.shape[0]
            a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform for each samples

            # loss matrix
            M = ot.dist(xs, xt)
            # M /= M.max()

            print(ot.emd2(a, b, M, numItermax = 1e+8))
            print(f'n is {n} and m is {m}')
            result_ot[i, j, h] = ot.emd2(a, b, M)

            # if result_ot[i, j] > 1000:
            #     abnormal_idx.append((i, j))

#%% box plot과 SME 같이..
import seaborn as sns
for h in [0]:
    # 0을 빼야함..
    result_ot_exclude_0 = []
    for i in range(LEN_AREA-1):
        result_tmp = []
        for j in range(LEN_AREA-1):
            if i == j:
                continue
            else:
                result_tmp.append(result_ot[i, j, h])
        result_ot_exclude_0.append(result_tmp)

    plt.figure(figsize=(8,4))
    sns.boxplot(data = result_ot_exclude_0)
    plt.plot(result_ot_exclude_0[-1], color='b', label='CER dataset')
    plt.xticks(range(LEN_AREA)[::3], range(LEN_AREA)[::3])
    plt.xlabel('index of dataset')
    plt.ylabel('Optimal distance')
    # plt.title('optimal distances b.t.w area')
    plt.title(f'h is {h}')
    plt.legend()
    plt.show()

#%% 집 하나를 골라서, 시간대별 분석
result_ot = np.zeros((LEN_AREA, 24))
for h in range(24):
    for i in range(LEN_AREA):
        xs = data_dict[area[0]][:100,h:h+1]
        xt = data_dict[area[i]][:100,h:h+1]

        n = xs.shape[0]
        m = xt.shape[0]
        a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform for each samples

        # loss matrix
        M = ot.dist(xs, xt)
        # M /= M.max()

        print(ot.emd2(a, b, M, numItermax = 1e+8))
        print(f'n is {n} and m is {m}')
        result_ot[i, h] = ot.emd2(a, b, M)

# %% boxplot
plt.figure(figsize=(8,4))
sns.boxplot(data = result_ot[:-1,:])
plt.plot(result_ot[-1], color='b', label='CER dataset')
# plt.xticks(range(LEN_AREA-1)[::3], range(LEN_AREA-1)[::3])
plt.xlabel('Hour')
plt.ylabel('OT distance')
plt.title('optimal distances b.t.w seoul and CER')
# plt.title(f'h is {h}')
plt.legend()
plt.show()


#%%
for i in range(LEN_AREA):
    plt.plot(data_dict[area[i]].mean(axis=0)[:24], color = 'k')
plt.plot(data_dict[area[i]].mean(axis=0)[:24], label = 'CER dataset', color = 'b')
plt.legend()
plt.title('지역별 평균 부하')
plt.xlabel('Hour')
plt.ylabel('Energy consumption [kW]')
plt.show()

#%%
plt.plot(xs.T, alpha = 0.8, color = 'k')
plt.plot(xt.T, alpha = 0.8,color = 'r')
plt.show()

#%% distribution
plt.figure(figsize=(6,4))
for i in range(LEN_AREA-1):
    plt.hist(data_dict[area[i]][:100,:].reshape(-1),
             label = area[i], alpha = 0.8, bins = 100, ls='solid', histtype='step',
             density=True)
plt.legend()
plt.show()

#%% 계절 영향 분석
result_ot = np.zeros((12))
for season in range(12):
    xs = data_dict[area[0]][:100,:]
    xt = data_dict['CER_R_'+str(season)][:100,:]

    n = xs.shape[0]
    m = xt.shape[0]
    a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform for each samples

    # loss matrix
    M = ot.dist(xs, xt)
    # M /= M.max()

    print(ot.emd2(a, b, M, numItermax = 1e+8))
    print(f'n is {n} and m is {m}')
    result_ot[season] = ot.emd2(a, b, M)


plt.plot(result_ot)
plt.show()
