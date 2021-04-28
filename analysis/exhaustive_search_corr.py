import multiprocessing
from module.util import *
import itertools

# inputu
CPU_CORE = multiprocessing.cpu_count()
k = 2

# load dataset
power_df = pd.read_csv('../data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
info = load_info(path='data/survey/survey_for_number_of_residents.csv')
power_df.set_index('time', inplace=True)
info[info>=7] = 7
label = info['count'].values
label = label.astype(int)

# for save result
possible_combs = list(itertools.combinations(np.arange(0,2*24), k))
results = np.zeros((len(possible_combs),2))

def calc_MI_corr(time_idx):
    ## MI
    print(time_idx)
    data = np.nanmean(power_df.copy().values[time_idx, :], axis=0)
    data[pd.isna(data)] = 0
    val, count = np.unique(label, return_counts=True)
    prob = count / count.sum()

    sig_ = np.std(data)
    H_X = 1 / 2 * np.log2(2 * np.pi * np.e * sig_ ** 2)

    H_con = 0
    for ii, v in enumerate(val):
        sig_ = np.std(data[label == v])
        H_con += (1 / 2 * np.log2(2 * np.pi * np.e * sig_ ** 2)) * prob[ii]
    MI = H_X - H_con

    ## corr
    corr_ = np.corrcoef(np.ravel(data), np.ravel(label))[0,1]
    print(corr_)
    return [MI, corr_]

for idx, time_idx in enumerate(possible_combs):
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=CPU_CORE)
        results[idx,:] = pool.map(calc_MI_corr, [list(time_idx)])
        pool.close()
        pool.join()