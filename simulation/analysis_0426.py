
# %% load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Save
SAVE = pd.read_csv('data/SAVE/power_0426.csv', index_col=0)
SAVE = SAVE.iloc[84:,:]
SAVE.index = pd.to_datetime(SAVE.index)
start_date = pd.to_datetime('2017-01-01 00:00:00')
end_date = pd.to_datetime('2017-03-01 23:00:00')
SAVE = SAVE.loc[start_date:end_date,:]

# CER
CER = pd.read_csv('data/CER/power_comb.csv', index_col=0)
CER.index = pd.to_datetime(CER.index)
start_date = pd.to_datetime('2010-01-01 00:00:00')
end_date = pd.to_datetime('2010-03-01 23:00:00')
CER = CER.loc[start_date:end_date,:]

# ETRI
ETRI = pd.read_csv('data/ETRI/label_0427.csv', index_col=0)
ETRI.index = pd.to_datetime(ETRI.index)
start_date = pd.to_datetime('2018-01-01 00:00:00')
end_date = pd.to_datetime('2018-03-01 23:00:00')
ETRI = ETRI.loc[start_date:end_date,:]

# %% load label
ETRI_label = pd.read_csv('data/ETRI/survey_0427.csv', index_col = 0)

CER_label = pd.read_csv('data/CER/survey_processed_0427.csv')
CER_label['ID'] = CER_label['ID'].astype(str)
CER_label.set_index('ID', inplace=True)
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

SAVE_label = pd.read_csv('data/SAVE/save_household_survey_updates_data_v0-3.csv', index_col = 0)

# %% Replace 0 with nan
SAVE[SAVE == 0] = np.nan
CER[CER == 0] = np.nan
ETRI[ETRI == 0] = np.nan

# %% downsampling and dim reduct
from module.util_main import downsampling, dim_reduct

# save
SAVE_rs, SAVE_id = dim_reduct(SAVE.values, 24 * 4, th = 0)
# SAVE_rs, SAVE_id = dim_reduct_v2(SAVE.values, 24 * 4, th = 0)
SAVE_id = SAVE.columns[SAVE_id]
SAVE_rs = downsampling(SAVE_rs, 4)

# cer
CER_rs, CER_id = dim_reduct(CER.values, 24 * 2, th = 0)
# CER_rs, CER_id = dim_reduct_v2(CER.values, 24 * 2, th = 0)
CER_id = CER.columns[CER_id]
CER_rs = downsampling(CER_rs, 2)

# etri
# ETRI_rs, ETRI_id = dim_reduct_v2(ETRI.values, 24, th = 0)
ETRI_rs, ETRI_id = dim_reduct(ETRI.values, 24, th = 0)
ETRI_id = ETRI.columns[ETRI_id]

# %% matching with label
CER_label = CER_label.loc[CER_id,:]
ETRI_label = ETRI_label.loc[:,ETRI_id]
SAVE_label.index = SAVE_label.index.astype(str)
i_list = []
SAVE_label_id_list = []
for i, id in enumerate(SAVE_id):
    if id[2:] in SAVE_label.index:
        i_list.append(i)
        SAVE_label_id_list.append(id[2:])
SAVE_rs = SAVE_rs[i_list,:]

SAVE_label = SAVE_label.loc[SAVE_label_id_list, :]
SAVE_label['id'] = SAVE_label.index
SAVE_label.drop_duplicates(subset=['id'], inplace = True, keep='first')
del SAVE_label['id']


#%% Distribution
plt.figure(figsize=(6, 3))
plt.hist(SAVE_rs.reshape(-1), bins = 100, alpha = 0.7, density = True, label = 'SAVE')
plt.hist(CER_rs.reshape(-1), bins = 100, alpha = 0.7, density = True, label = 'CER')
plt.hist(ETRI_rs.reshape(-1), bins = 100, alpha = 0.7, density = True, label = 'ETRI')
plt.xlim([0, 4])
plt.xlabel('Energy consumption [kWh]')
plt.ylabel('Density')
plt.title('PDF')
plt.legend()
plt.show()

# %% shape 확인
print(CER_rs.shape)
print(CER_label.shape)

print(SAVE_rs.shape)
print(SAVE_label.shape)

print(ETRI_rs.shape)
print(ETRI_label.shape)

# %% correlation with 사람수
nan_idx = np.any(pd.isnull(CER_rs), axis=1)
CER_rs = CER_rs[~nan_idx, :]
CER_label = CER_label.iloc[~nan_idx, :]

nan_idx = np.any(pd.isnull(SAVE_rs), axis=1)
SAVE_rs = SAVE_rs[~nan_idx, :]
SAVE_label = SAVE_label.iloc[~nan_idx, :]

nan_idx = np.any(pd.isnull(ETRI_rs), axis=1)
ETRI_rs = ETRI_rs[~nan_idx, :]
ETRI_label = ETRI_label.T.iloc[~nan_idx, :]

# %%
from collections import defaultdict
corr = defaultdict(list)
for i in range(24):
    corr['ETRI'].append(np.corrcoef(ETRI_rs[:,i], ETRI_label['popl_num'].values)[0,1])
    corr['CER'].append(np.corrcoef(CER_rs[:,i], CER_label['Q13'].values)[0,1])
    corr['SAVE'].append(np.corrcoef(SAVE_rs[:,i], SAVE_label['Q2'].values)[0,1])

# %%
plt.figure(figsize =(6,3))
plt.plot(corr['ETRI'], label = 'ETRI')
plt.plot(corr['CER'], label = 'CER')
plt.plot(corr['SAVE'], label = 'SAVE')
plt.ylabel('Correlation')
plt.xlabel('Hour')
plt.legend()
plt.show()

# %% 유사도
import ot

xs = CER_rs[:200,:]
xt = SAVE_rs[:200,:]

n = xs.shape[0]
m = xt.shape[0]
a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform for each samples

# loss matrix
M = ot.dist(xs, xt)
# M /= M.max()

print(ot.emd2(a, b, M, numItermax = 1e+8))

