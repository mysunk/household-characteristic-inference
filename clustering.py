

# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% load dataset
start_date = pd.to_datetime('2009-09-01 00:00:00')
end_date = pd.to_datetime('2009-12-01 23:00:00')

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
    house_idx.append(np.ones(data.shape[0], dtype=int) * i)
print('Done')

data_all = np.concatenate(data_all, axis=0)
house_idx = np.concatenate(house_idx, axis=0)

# downsampling
from module.util_main import downsampling
data_all = downsampling(data_all)

# %% Daily clustering
clusterig_result = dict()
model_dict = dict()
from tslearn.clustering import TimeSeriesKMeans, KShape

for n_clusters in tqdm(range(2, 10)):
    # 1. DTW with kmeans
    model = TimeSeriesKMeans(n_clusters = n_clusters, metric = 'dtw', \
        max_iter = 500, random_state = 0, verbose = 1, n_jobs=-1)
    result = model.fit_predict(data_all)
    model_dict['dtw_' + str(n_clusters)] = model
    clusterig_result['dtw_' + str(n_clusters)] = result

    # 2. KMeans
    model = TimeSeriesKMeans(n_clusters = n_clusters, metric = 'euclidean', \
        max_iter = 500, random_state = 0, verbose = 1, n_jobs=-1)
    result = model.fit_predict(data_all)
    model_dict['euc_' + str(n_clusters)] = model
    clusterig_result['euc_' + str(n_clusters)] = result

    # 3. KShape
    model = KShape(n_clusters = n_clusters, max_iter = 500, verbose = 1, random_state = 0)
    result = model.fit_predict(data_all)
    model_dict['kshape_' + str(n_clusters)] = model
    clusterig_result['kshape_' + str(n_clusters)] = result


# %% result
n_clusters = 2
method = 'euc_'+str(n_clusters)
result= clusterig_result[method]
model= model_dict[method]
for i in range(n_clusters):
    data = data_all[result== i]
    plt.plot(data.T, color = 'k', alpha = 0.1)
    plt.plot(model.cluster_centers_[i], label = 'centroid')
    plt.title('Cluster {}'.format(i))
    plt.xlabel('Hour')
    plt.ylabel('Energy consumption [kWh]')
    plt.legend()
    plt.show()

# %% DB index 확인
from sklearn.metrics import davies_bouldin_score
db_result_dtw = []
db_result_kmn = []
for n_clusters in range(2, 10):
    # db_result_dtw.append(davies_bouldin_score(data_all, \
    #     clusterig_result['dtw_'+str(n_clusters)]))
    db_result_kmn.append(davies_bouldin_score(data_all, \
        clusterig_result['euc_'+str(n_clusters)]))

plt.figure(figsize=(6,3))
# plt.plot(range(2, 10), db_result_dtw, label='DTW')
plt.plot(range(2, 10), db_result_kmn, label='KMeans')
plt.title('DB index')
plt.xlabel('# of clusters')
plt.legend()
plt.show()


# %%
# clustering 결과
result_ref.shape
house_idx


# %%
result_ref = clusterig_result['euc_' + str(n_clusters)] 
ref_pc = dict()
selected_data = dict()
for i in range(100):
    idx = house_idx == i
    v, c = np.unique(result_ref[idx], return_counts = True)
    v_selected = v[np.argmax(c)]
    ref_pc[i] = v_selected 
    selected_data[i] = data_all[idx][result_ref[idx] == v_selected].mean(axis=0)
    # selected_data[i] = data_all[result_ref == v_selected].mean(axis=0)


# %%
sample_idx = 0
plt.plot(np.mean(data_all[house_idx == sample_idx], axis=0), label='3 month avg')
plt.plot(selected_data[sample_idx], label='same cluster avg')
plt.legend()
plt.show()

# %%

