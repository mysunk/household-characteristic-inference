import pandas as pd
import numpy as np
import copy

from scipy.interpolate import interp1d
from datetime import datetime, timedelta


def Process_F_Num(file_num, path = 'data/power_raw/'):
    """
    Separate CER dataset File 1 ~6 . txt into .csv for each smart meter.


    Parameters
    ----------
    file_num : integer, from 1 to 6.


    Returns
    -------
    None.

    """
    F_name = 'File' + str(file_num) + '.txt'

    df = pd.read_csv(path + F_name, sep=" ", names=["meter_id", "index", "usage"])

    meter_names = df['meter_id'].unique()

    cnt_normal = 0
    cnt_abnorm = 0
    for meters in meter_names:
        print(meters)
        temp_df = copy.deepcopy(df[df['meter_id'] == meters])
        temp_df.pop('meter_id')
        start_day = temp_df['index'].min() // 100
        end_day = temp_df['index'].max() // 100
        duration = end_day - start_day + 1
        temp_df = temp_df.sort_values(by=['index'])

        temp_df = over48(temp_df)
        print(pd.isnull(temp_df).sum())
        temp_df = fillgap_interp(temp_df)

        if temp_df.shape[0] / 48 == duration:
            temp_df.to_csv(str(meters) + '_' + str(start_day) + '_' +
                           str(duration) + '_' + str(end_day) + '.csv',
                           mode='w')
            cnt_normal = cnt_normal + 1
        else:
            temp_df.to_csv('c' + str(meters) + '_' + str(start_day) + '_' +
                           str(duration) + '_' + str(end_day) + '(' + str(temp_df.shape[0] // 48) + ').csv',
                           mode='w')
            cnt_abnorm = cnt_abnorm + 1

    print(str(cnt_normal) + ' normal and ' + str(cnt_abnorm) + ' abnormal are separated')
    print('Total ' + str(cnt_normal + cnt_abnorm))


def over48(dataframe):
    ar = np.array(dataframe)
    ar = np.delete(ar, np.where(ar[:, 0] % 100 > 48), 0)

    df = pd.DataFrame(ar, columns=['index', 'usage'])
    return df


def fillgap_interp(dataframe):
    ar = np.array(dataframe)
    startdate = datetime(2009, 1, 1) + timedelta(days=ar[0, 0] // 100 - 1)
    enddate = datetime(2009, 1, 1) + timedelta(days=ar[-1, 0] // 100 - 1) - timedelta(minutes=30)

    t = pd.date_range(startdate, enddate + timedelta(days=1), freq="30T")
    t = pd.to_datetime(t)

    duration = (enddate - startdate).days + 2
    val_ar = np.ones([duration, 48]) * -1
    for i in range(duration):
        for ii in range(48):
            idx = (ar[0, 0] // 100 + i) * 100 + ii + 1
            if ar[np.where(ar[:, 0] == idx), 1].size == 1:
                val_ar[i, ii] = ar[np.where(ar[:, 0] == idx), 1]
            elif ar[np.where(ar[:, 0] == idx), 1].size > 1:
                val_ar[i, ii] = np.mean(ar[np.where(ar[:, 0] == idx), 1])
    print((val_ar == -1).sum())
    val_vec = val_ar.reshape([val_ar.size, 1])
    val_vec[np.where(val_vec == -1)] = float('nan')
    Se_out = pd.Series(val_vec[:, 0], index=t)
    Se_out.interpolate()

    return Se_out


if __name__ == '__main__':
    Process_F_Num(1)