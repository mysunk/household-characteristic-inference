import numpy as np
import pandas as pd


def load_info(path):
    info = pd.read_csv(path)
    info.set_index('ID', inplace=True)

    # replace blanks
    for j in range(info.shape[1]):
        for i in range(info.shape[0]):
            try:
                int(info.iloc[i, j])
            except:
                info.iloc[i, j] = -1

    for j in range(info.shape[1]):
        info.iloc[:, j] = info.iloc[:, j].astype(int)

    info['count'] = np.zeros((info.shape[0]), dtype=int)
    info['children'] = np.zeros((info.shape[0]), dtype=int)

    # 세대원 수가 1명인 경우
    info.loc[info['Q410'] == 1, 'count'] = 1
    info.loc[info['Q410'] == 1, 'children'] = 0

    # 모든 세대원이 15살 이상인 경우
    info.loc[info['Q410'] == 2, 'count'] = info.loc[info['Q410'] == 2, 'Q420']
    info.loc[info['Q410'] == 2, 'children'] = 0

    # children이 있는 경우
    info.loc[info['Q410'] == 3, 'count'] = info.loc[info['Q410'] == 3, 'Q420'] + info.loc[info['Q410'] == 3, 'Q43111']
    info.loc[info['Q410'] == 3, 'children'] = info.loc[info['Q410'] == 3, 'Q43111']
    return info

def calc_MI_corr(data, label):
    ## MI
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
    return [MI, corr_]