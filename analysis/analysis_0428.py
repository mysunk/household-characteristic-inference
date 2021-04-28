# %% load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Save
SAVE = pd.read_csv('data/SAVE/power_0428.csv', index_col=0)
SAVE = SAVE.iloc[84:,:]
SAVE.index = pd.to_datetime(SAVE.index)
SAVE[SAVE == 0] = np.nan

SAVE_label = pd.read_csv('data/SAVE/save_household_survey_updates_data_v0-3.csv', index_col = 0)

# %%
# 계절별로 나눔
SAVE_data_dict = dict()

start_date = pd.to_datetime('2017-01-01 00:00:00')
end_date = pd.to_datetime('2017-02-15 23:00:00')
SAVE_data_dict['겨울'] = SAVE.loc[start_date:end_date,:]

start_date = pd.to_datetime('2017-03-01 00:00:00')
end_date = pd.to_datetime('2017-04-15 23:00:00')
SAVE_data_dict['봄'] = SAVE.loc[start_date:end_date,:]

start_date = pd.to_datetime('2017-05-01 00:00:00')
end_date = pd.to_datetime('2017-06-15 23:00:00')
SAVE_data_dict['여름'] = SAVE.loc[start_date:end_date,:]


# %% 계절별로 데이터를 나눔
from module.util_main import downsampling, dim_reduct
from collections import defaultdict

plt.figure(figsize =(6,3))

keys = ['겨울', '봄', '여름']
corr = defaultdict(list)
for key in keys:
    data = SAVE_data_dict[key]
    SAVE_rs, SAVE_id = dim_reduct(data.values, 24 * 4, th = 0)
    SAVE_id = data.columns[SAVE_id]
    SAVE_rs = downsampling(SAVE_rs, 4)

    # matching with label
    i_list = []
    SAVE_label_id_list = []
    for i, id in enumerate(SAVE_id):
        if id[2:] in SAVE_label.index.astype(str):
            i_list.append(i)
            SAVE_label_id_list.append(id[2:])
    
    SAVE_rs = SAVE_rs[i_list,:]
    SAVE_label_tmp = SAVE_label.loc[SAVE_label_id_list, :]
    SAVE_label_tmp['id'] = SAVE_label_tmp.index
    SAVE_label_tmp.drop_duplicates(subset=['id'], inplace = True, keep='first')

    for i in range(24):
        corr[key].append(np.corrcoef(SAVE_rs[:,i], SAVE_label_tmp['Q2'].values)[0,1])
    plt.plot(corr[key], label = key)

plt.ylabel('Correlation')
plt.xlabel('Hour')
plt.legend()
plt.show()

# %% nan 분포 확인
plt.plot(pd.isnull(SAVE).sum(axis=1))
plt.xticks(rotation = 30)
plt.show()
