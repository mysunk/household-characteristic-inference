import matplotlib
from module.util import *
font = {'size': 16}
matplotlib.rc('font', **font)

# 데이터 로드 및 전처리
power_df = pd.read_csv('../data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
info = load_info(path='data/survey_for_number_of_residents.csv')
datetime = power_df['time']
datetime = pd.to_datetime(datetime)
power_df.set_index('time', inplace=True)

def get_profile(daytype):
    """
    Parameters
    ----------
    daytype: 1: 전체 2: 주중 3: 주말
    Returns: transformed energy consumption data
    -------
    """
    power_arr = power_df.copy().values
    weekday_flag = np.reshape(datetime.dt.weekday.values, (-1, 48))[:, 0]
    if daytype == 1:
        weekday_flag[:] = True
    elif daytype == 2:
        weekday_flag = [w < 5 for w in weekday_flag]
    elif daytype == 3:
        weekday_flag = [w >= 5 for w in weekday_flag]

    weekday_flag = np.array(weekday_flag).astype(bool)
    power_arr_rs = np.reshape(power_arr.T, (power_df.shape[1], -1, 48))  # home, day, hours
    power_arr_rs = power_arr_rs[:, weekday_flag, :]
    power_arr_rs = np.nanmean(power_arr_rs, axis=1)
    return power_arr_rs

#%% (1) consumption feature
features = []

# 1. mean P, daily, all day
feature = get_profile(daytype = 1)
feature = np.mean(feature, axis=1).reshape(-1,1)
features.append(feature)

# 2. mean P, daily, weekday
feature = get_profile(daytype = 2)
feature = np.mean(feature, axis=1).reshape(-1,1)
features.append(feature)

# 3. mean P, daily, weekend
feature = get_profile(daytype = 3)
feature = np.mean(feature, axis=1).reshape(-1,1)
features.append(feature)

# 4. mean P, 6 ~ 22
feature = get_profile(daytype = 1)
feature = np.mean(feature[:,2*6:2*22], axis=1).reshape(-1,1)
features.append(feature)

# 5. mean P, 6 ~ 8.5
feature = get_profile(daytype = 1)
feature = np.mean(feature[:,2*6:2*8+1], axis=1).reshape(-1,1)
features.append(feature)

# 6. mean P, 8.5 ~ 12
feature = get_profile(daytype = 1)
feature = np.mean(feature[:,2*8+1:2*12], axis=1).reshape(-1,1)
features.append(feature)

# 7. mean P, 12 ~ 14.5
feature = get_profile(daytype = 1)
feature = np.mean(feature[:,2*12:2*14+1], axis=1).reshape(-1,1)
features.append(feature)

# 7. mean P, 14.5 ~ 18
feature = get_profile(daytype = 1)
feature = np.mean(feature[:,2*14+1:2*18], axis=1).reshape(-1,1)
features.append(feature)