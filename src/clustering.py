import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% load dataset
start_date = pd.to_datetime('2009-07-14 00:00:00')
end_date = pd.to_datetime('2009-08-14 23:00:00')

src_path = 'D:/GITHUB/python_projects/CER-information-inference/'
power_df_raw = pd.read_csv(src_path + 'data/power_comb_SME_included.csv')

# 0 to NaN
power_df_raw[power_df_raw==0] = np.nan
power_df_raw['time'] = pd.to_datetime(power_df_raw['time'])
power_df_raw.index = power_df_raw['time']
power_df = power_df_raw.loc[start_date : end_date,:]
power_df.set_index('time', inplace=True)

# SME & residential
code = pd.read_excel(src_path + 'data/doc/SME and Residential allocations.xlsx', engine='openpyxl')
residential = code.loc[code['Code'] == 1,'ID'].values
SME = code.loc[code['Code'] == 2,'ID'].values
power_residential = power_df[residential.astype(str)]
power_sme = power_df[SME.astype(str)]

#%% data 쌓기
window = 24 *2
# n_house = power_residential.shape[1]
n_house = 100
house_idx = []
data_all = []
for i in tqdm(range(n_house)):
    data = power_residential.iloc[:,i].values
    cut = (data.shape[0] // window) * window
    data = data[:cut]
    data = data.reshape(-1, window) # weekly

    # drop nan
    nan_idx = np.any(pd.isnull(data), axis=1)
    data = data[~nan_idx,:]
    data_all.append(data)
    house_idx.append(np.ones(data.shape[0], dtype=int))
print('Done')

data_all = np.concatenate(data_all, axis=0)
house_idx = np.concatenate(house_idx, axis=0)

# %% Daily clustering
clusterig_result = dict()
from tslearn.clustering import TimeSeriesKMeans
for n_clusters in tqdm(range(2, 20)):
    model = TimeSeriesKMeans(n_clusters = n_clusters, metric = 'dtw', \
        max_iter = 500, random_state = 0, verbose = 0, n_jobs=-1)
    result = model.fit_predict(data_all)
    model_ref = TimeSeriesKMeans(n_clusters = n_clusters, metric = 'euclidean', \
        max_iter = 500, random_state = 0, verbose = 0, n_jobs=-1)
    result_ref = model_ref.fit_predict(data_all)

    clusterig_result['dtw_' + str(n_clusters)] = result
    clusterig_result['euc_' + str(n_clusters)] = result_ref

#%% 
# DTW
model = TimeSeriesKMeans(n_clusters = 6, metric = 'dtw', \
max_iter = 1000, random_state = 0, verbose = 1, n_jobs=-1)
result = model.fit_predict(data_all)

# kmeans
model_ref = TimeSeriesKMeans(n_clusters = 6, metric = 'euclidean', \
max_iter = 1000, random_state = 0, verbose = 1, n_jobs=-1)
result_ref = model_ref.fit_predict(data_all)

# %% 
result = clusterig_result['euc_19']
for i in range(19):
    data = data_all[result == i]
    plt.plot(data.T, color = 'k', alpha = 0.1)
    plt.plot(model_ref.cluster_centers_[i], label = 'centroid')
    plt.plot(median_, color = 'r', label='median')
    plt.title('Cluster {}'.format(i))
    plt.legend()
    plt.show()

# %% DB index 확인
from sklearn.metrics import davies_bouldin_score
db_result_dtw = []
db_result_kmn = []
for n_clusters in range(2, 20):
    db_result_dtw.append(davies_bouldin_score(data_all, \
        clusterig_result['dtw_'+str(n_clusters)]))
    db_result_kmn.append(davies_bouldin_score(data_all, \
        clusterig_result['euc_'+str(n_clusters)]))


plt.plot(range(2, 20), db_result_dtw)
plt.plot(range(2, 20), db_result_kmn)
plt.show()


# %% PCA를 통한 clustering결과 확인

