import matplotlib
from module.util import *
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

# 데이터 로드 및 전처리
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
power_df['time'] = pd.to_datetime(power_df['time'])
weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)

#%%
