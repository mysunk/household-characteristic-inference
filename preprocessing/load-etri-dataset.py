
# %%
import pandas as pd
import numpy as np

# %% load energy consumption data

data_path = 'D:/ISP/3. 데이터/ENERGY/ETRI/'
data_raw = pd.read_csv(data_path + 'label_data.csv', low_memory=False, dtype={
        'Time': str,
        'Season': str,
        'Weekday': str,
    })

data_raw.index = pd.to_datetime(data_raw['Time'])
# data_raw = data_raw.loc[start_date:end_date, :]
data_raw[(data_raw.values == 0)] = np.nan

# extra info 데이터와 순서를 맞춤
people_n = pd.read_excel('data/ETRI/people_num.xlsx', header=None, index_col=0)
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
# data = data.loc[start_date:end_date, :].copy()
data.drop(columns=['Time', 'Season', 'Weekday'], inplace=True)

# %% load survey data
idx = data.columns

people_n = pd.read_excel('data/ETRI/people_num.xlsx', header=None, index_col=0)
appliance = pd.read_excel('data/ETRI/appliance.xlsx', index_col=0, header=None)
working_info = pd.read_excel('data/ETRI/working_info.xlsx', index_col=0, header=None)

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

extra_info.loc['appl_num',:] = extra_info.loc['ap_1':'ap_11',:].sum(axis=0)
extra_info.loc['popl_num', :] = extra_info.loc['m_0':'w_70', :].sum(axis=0)
extra_info.loc['adult_num', :] = (extra_info.loc['m_30':'m_70', :].values + extra_info.loc['w_30':'w_70', :].values).sum(axis=0)
extra_info.loc['erderly_num', :] = (extra_info.loc['m_60':'m_70', :].values + extra_info.loc['w_60':'w_70', :].values).sum(axis=0)
extra_info.loc['child_num', :] = extra_info.loc['m_0', :].values + extra_info.loc['w_0', :].values
extra_info.loc['teen_num', :] = extra_info.loc['m_10', :].values + extra_info.loc['w_10', :].values
extra_info.loc['child_include_teen', :] = (extra_info.loc['m_0':'m_10', :].values + extra_info.loc['w_0':'w_10', :].values).sum(axis=0)
extra_info.loc['male_num', :] = extra_info.loc['m_0':'m_70', :].sum(axis=0)
extra_info.loc['female_num', :] = extra_info.loc['w_0':'w_70', :].sum(axis=0)
extra_info.loc['income_solo', :] = (extra_info.loc['income_type', :] == 2).astype(int)
extra_info.loc['income_dual', :] = (extra_info.loc['income_type', :] == 1).astype(int)
extra_info.loc['work_home', :] = (extra_info.loc['work_type', :] == 1).astype(int)
extra_info.loc['work_office', :] = (extra_info.loc['work_type', :] == 2).astype(int)
extra_info.drop(index=extra_info.index[12:28], inplace=True)
extra_info.drop(index=['income_type','work_type'], inplace=True)
extra_info = extra_info.astype(int)


# %% save data
data.to_csv('data/ETRI/label_0427.csv')
extra_info.to_csv('data/ETRI/survey_0427.csv')
