# %% load raw dataset
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join


data_list = []
SOURCE_PATH = 'D:/ISP/3. 데이터/ENERGY/SAVE/csv/'
# PATH_LIST = ['save_consumption_data_2017_1_v0-1/', 'save_consumption_data_2017_2_v0-1/', \
#             'save_consumption_data_2018_1_v0-1/', 'save_consumption_data_2018_2_v0-1/']
PATH_LIST = ['save_consumption_data_2017_1_v0-1/']
for path in PATH_LIST:
    PATH = SOURCE_PATH + path
    file_list = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    for file in file_list:
        data = pd.read_csv(PATH+file, low_memory=False)
        data_list.append(data)

# concatenate
data = pd.concat(data_list, axis=0)
# residential ID list
bmg_id_list = np.unique(data['bmg_id'])

# %% Make dataframe
# empty dataframe
from datetime import datetime, timedelta
date = data['recorded_timestamp']
date = np.unique(date)
date = pd.Series(date)
date = date.apply(lambda d: datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M'))
df = pd.DataFrame(index = date)

# fill the empty dataframe
from tqdm import tqdm
for bmg_id in tqdm(bmg_id_list):
    idx = data['bmg_id'] == bmg_id
    time = data.loc[idx, 'recorded_timestamp']
    time = time.apply(lambda d: datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M'))
    energy = data.loc[idx,'energy']
    if energy.shape[0] == 1:
        continue
    energy = energy.diff()
    energy.iloc[0] = energy.iloc[1] # fill nan
    df[bmg_id] = np.nan
    df.loc[time, bmg_id] = energy.values

# %% Wh to kWh
df = df / 1000
df.to_csv('data/SAVE/power_0426.csv', index = True)


# %% survey
file_path = 'D:/ISP/3. 데이터/ENERGY/SAVE/csv/save_household_survey_data/save_household_survey_updates_data_v0-3.csv'
survey = pd.read_csv(file_path)
