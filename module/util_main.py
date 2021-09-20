import pandas as pd
import numpy as np


def dimension_reduction(data, window = 24 * 7):
    '''
    make 3d data to 2d
    '''
    data_3d = data.reshape(-1, window, data.shape[1])  # day, hour, home
    data_3d = data_3d.transpose(2, 0, 1)
    data_2d = np.nanmean(data_3d, axis=1)
    return data_2d


def data_aug(data, window = 24 * 7):
    '''
    data를 window 단위로 nan이 있는 주를 제외하함
    '''
    cut = (data.shape[0] // window) * window
    data = data[:cut, :]

    data_3d = data.reshape(-1, window, data.shape[1])  # day, hour, home
    data_3d = data_3d.transpose(2, 0, 1)
    data_3d = data_3d.reshape(-1, window)
    nan_day = np.any(pd.isnull(data_3d), axis=1)
    data_2d = data_3d[~nan_day, :]  # nan이 없는 집 삭제
    return data_2d


def load_energy(data_p, start_date, end_date, additional=None):
    data_path = 'D:/GITHUB/python_projects/etri-load-preprocessing/'
    data_raw = pd.read_csv(data_p, low_memory=False, dtype={
        'Time': str,
        'Season': str,
        'Weekday': str,
    })
    data_raw.index = pd.to_datetime(data_raw['Time'])
    data_raw = data_raw.loc[start_date:end_date, :]
    data_raw[(data_raw.values == 0)] = np.nan
    if not additional:
        data_raw.drop(columns=['Time', 'Season', 'Weekday'], inplace=True)
        return data_raw

    # extra info 데이터와 순서를 맞춤
    people_n = pd.read_excel(data_path + 'data/people_num.xlsx', header=None, index_col=0)
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
            continue
    data = data.loc[start_date:end_date, :].copy()
    data.drop(columns=['Time', 'Season', 'Weekday'], inplace=True)
    return data


def load_extra_info(idx):
    data_path = 'D:/GITHUB/python_projects/etri-load-preprocessing/'
    people_n = pd.read_excel(data_path + 'data/people_num.xlsx', header=None, index_col=0)
    appliance = pd.read_excel(data_path + 'data/appliance.xlsx', index_col=0, header=None)
    working_info = pd.read_excel(data_path + 'data/working_info.xlsx', index_col=0, header=None)

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
                        'ap_7', 'ap_8', 'ap_9', 'ap_10', 'ap_11',
                        'm_0', 'm_10', 'm_20', 'm_30', 'm_40', 'm_50', 'm_60', 'm_70',
                        'w_0', 'w_10', 'w_20', 'w_30', 'w_40', 'w_50', 'w_60', 'w_70',
                        'income_type', 'work_type']
    extra_info.loc['appl_num', :] = extra_info.loc['ap_1':'ap_11', :].sum(axis=0)
    extra_info.loc['popl_num', :] = extra_info.loc['m_0':'w_70', :].sum(axis=0)
    extra_info.loc['adult_num', :] = (
            extra_info.loc['m_30':'m_70', :].values + extra_info.loc['w_30':'w_70', :].values).sum(axis=0)
    extra_info.loc['erderly_num', :] = (
            extra_info.loc['m_60':'m_70', :].values + extra_info.loc['w_60':'w_70', :].values).sum(axis=0)
    extra_info.loc['child_num', :] = extra_info.loc['m_0', :].values + extra_info.loc['w_0', :].values
    extra_info.loc['teen_num', :] = extra_info.loc['m_10', :].values + extra_info.loc['w_10', :].values
    extra_info.loc['child_include_teen', :] = (
            extra_info.loc['m_0':'m_10', :].values + extra_info.loc['w_0':'w_10', :].values).sum(axis=0)
    extra_info.loc['male_num', :] = extra_info.loc['m_0':'m_70', :].sum(axis=0)
    extra_info.loc['female_num', :] = extra_info.loc['w_0':'w_70', :].sum(axis=0)
    extra_info.loc['income_solo', :] = (extra_info.loc['income_type', :] == 2).astype(int)
    extra_info.loc['income_dual', :] = (extra_info.loc['income_type', :] == 1).astype(int)
    extra_info.loc['work_home', :] = (extra_info.loc['work_type', :] == 1).astype(int)
    extra_info.loc['work_office', :] = (extra_info.loc['work_type', :] == 2).astype(int)
    extra_info.drop(index=extra_info.index[12:28], inplace=True)
    extra_info.drop(index=['income_type', 'work_type'], inplace=True)
    extra_info = extra_info.astype(int)
    return extra_info


def load_ETRI(option = 'target'):
    data_source_p = 'D:/ISP/8. 과제/2020 ETRI/data/SG_data_서울_비식별화/1147010100.csv'
    data_target_p = 'D:/ISP/8. 과제/2020 ETRI/data/label_data.csv'

    # source data
    if option == 'source':
        start_date = pd.to_datetime('2018-07-14 00:00:00')
        end_date = pd.to_datetime('2019-07-14 23:00:00')
        data_source = load_energy(data_source_p, start_date, end_date)
        data_source = data_source.values
        window = 24 * 7
        data_2d = data_aug(data_source, window)
        return data_2d
    else:
        start_date = pd.to_datetime('2018-07-14 00:00:00')
        end_date = pd.to_datetime('2018-08-10 23:00:00')

        # target data
        data_target = load_energy(data_target_p, start_date, end_date, additional=True)
        # label
        extra_info = load_extra_info(idx=data_target.columns).astype(int)
        data_target = dimension_reduction(data_target.values)
        nan_home = np.any(pd.isnull(data_target), axis=1)
        data_target = data_target[~nan_home, :]
        extra_info = extra_info.iloc[:, ~nan_home]

        # 1. number of residents
        label_residents = extra_info.loc['popl_num', :].values.copy()
        label_residents[label_residents <= 2] = 0
        label_residents[label_residents > 2] = 1

        # 2. number of appliances
        label_appliances = extra_info.loc['appl_num', :].values.copy()
        label_appliances[label_appliances <= 6] = 0
        label_appliances[(label_appliances > 6) * (label_appliances <= 8)] = 1
        label_appliances[label_appliances > 8] = 2

        # 3. have child
        label_child = extra_info.loc['child_num', :].values.copy()
        label_child[label_child > 0] = 1

        # 4. have child include teen
        label_child_w_teen = extra_info.loc['child_include_teen', :].values.copy()
        label_child_w_teen[label_child_w_teen > 0] = 1

        # 5. single
        label_single = (extra_info.loc['adult_num', :].values == 1) * (extra_info.loc['child_include_teen', :].values == 0)
        label_single = label_single.astype(int)

        # 6. area
        label_area = extra_info.loc['area', :].values.copy()
        label_area[label_area < 20] = 0
        label_area[(label_area >= 20) * (label_area <= 22)] = 1
        label_area[label_area > 22] = 2

        label_dict = dict()
        label_dict['Q1'] = label_residents
        label_dict['Q2'] = label_appliances
        label_dict['Q3'] = label_child
        label_dict['Q4'] = label_child_w_teen
        label_dict['Q5'] = label_single
        label_dict['Q6'] = label_area

        # ETRI와 같은 형식으로 맞춰주기 위해
        data_dict = dict()
        for i in range(1, 7):
            data_dict['Q' + str(i)] = data_target

        return data_dict, label_dict

def downsampling(data_tmp, interval):
    '''
    # 48 point를 24 point로 down sampling # fixme
    '''
    data_tmp_down = []
    for i in range(0, data_tmp.shape[1], interval):
        max_val = np.sum(data_tmp[:, i:i + interval], axis=1).reshape(-1, 1)
        data_tmp_down.append(max_val)
    data_tmp_down = np.concatenate(data_tmp_down, axis=1)
    return data_tmp_down

def load_CER(option = 'target'):
    ## load smart meter data
    power_df = pd.read_csv('D:/GITHUB/python_projects/CER-information-inference/data/power_comb.csv')
    # 0 to NaN
    power_df[power_df == 0] = np.nan
    window = 2*24*7
    power_df['time'] = pd.to_datetime(power_df['time'])
    weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
    power_df.set_index('time', inplace=True)

    if option == 'source':
        data_tmp = power_df.values
        data_2d = data_aug(data_tmp, window = 2 * 24 * 7)
        data_2d_down = downsampling(data_2d)
        return data_2d_down
    else:
        survey = pd.read_csv('D:/GITHUB/python_projects/CER-information-inference/data/survey_processed_0222.csv')
        survey['ID'] = survey['ID'].astype(str)
        survey.set_index('ID', inplace=True)

        power_df = power_df.iloc[:window * 4, :]  # 4주
        power_dict = dict()
        power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48 * 7))
        non_nan_idx = np.all(~pd.isnull(power_rs), axis=2)
        a = []
        for i, key in enumerate(power_df.columns):
            if non_nan_idx.sum(axis=1)[i] >= 2:  # 2주 이상 있음..
                power_dict[key] = []
                # all day
                power_dict[key].append(power_rs[i, non_nan_idx[i], :])

        def matching_ids(QID):
            # make label corresponding QID
            label_raw = survey[QID]
            label_raw = label_raw[label_raw != -1]

            data = []
            label = []
            for id in label_raw.index:
                id = str(id)
                if id in power_dict.keys():
                    # 24시간 프로파일
                    data_tmp = np.mean(power_dict[id][0], axis=0).reshape(1, -1)
                    data_tmp_down = downsampling(data_tmp)
                    data.append(data_tmp_down)
                    label.append(label_raw.loc[id])
            data = np.concatenate(data, axis=0)
            label = np.array(label)
            return data, label

        def matching_ids_with_aug(QID):
            # make label corresponding QID
            label_raw = survey[QID]
            label_raw = label_raw[label_raw != -1]

            data = []
            label = []
            for id in label_raw.index:
                id = str(id)
                if id in power_dict.keys():
                    # 24시간 프로파일
                    non_nan_idx = np.all(~pd.isnull(power_dict[id][0]), axis=1)
                    data_tmp = power_dict[id][0][non_nan_idx, :]
                    data_tmp = data_tmp[:2, :]  # 2주만 가져옴
                    data.append(data_tmp)
                    label.append(np.repeat(label_raw.loc[id], 14))
            data = np.concatenate(data, axis=0)
            label = np.ravel(np.array(label))
            return data, label

        data_dict, label_dict, data_ref_dict, label_ref_dict = dict(), dict(), dict(), dict()
        for i in range(1, 16):
            data_ref, label_ref = matching_ids('Q' + str(i))
            # data, label = matching_ids_with_aug('Q' + str(i))

            label_ref_dict['Q' + str(i)] = label_ref
            data_ref_dict['Q' + str(i)] = data_ref

            # data_dict['Q' + str(i)] = data
            # label_dict['Q' + str(i)] = label

        # return data_dict, label_dict
        return data_ref_dict, label_ref_dict


import matplotlib
font = {'size': 16}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
def plot_history(histories, key='loss'):
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')
        if key == 'loss':
            idx = np.argmin(history.history['val_' + key])
        else:
            idx = np.argmax(history.history['val_' + key])
        best_tr = history.history[key][idx]
        best_val = history.history['val_' + key][idx]

        print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr, key, best_val))

    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.xlim([0, max(history.epoch)])


def plot_learning(question_list, histories, GNT_list, metric, N_TRIAL):
    for QUESTION in question_list.keys():
        losses, acces = [], []
        for i in range(N_TRIAL):
            loss = np.min(histories[QUESTION][i].history['val_loss'])
            acc = np.max(histories[QUESTION][i].history['val_accuracy'])
            losses.append(loss)
            acces.append(acc)
        if metric == 'loss':
            best_history = np.argmin(losses)
        else:
            best_history = np.argmax(acces)
        history = histories[QUESTION][best_history]
        baseline = np.max(GNT_list[QUESTION].mean(axis=0))
        plt.subplot(2, 1, 1)
        if metric == 'loss':
            idx = np.argmin(history.history['val_loss'])
        else:
            idx = np.argmax(history.history['val_accuracy'])

        plt.title('{}({}) loss:{:.2f}, acc:{:.2f} '.format(QUESTION, question_list[QUESTION],
                                                           history.history['val_loss'][idx],
                                                           history.history['val_accuracy'][idx]))
        plot_history([('', history)], key='loss')
        plt.subplot(2, 1, 2)
        plot_history([('', history)], key='accuracy')
        plt.axhline(y=baseline, color='r', linestyle=':', label='baseline')
        plt.legend()
        plt.show()


def plot_pred_result(question_list, histories, question_pred_result, GNT_list, option, metric,  EPOCH_SIZE, N_TRIAL):
    from sklearn.decomposition import PCA

    for QUESTION in question_list.keys():

        pred_result = question_pred_result[QUESTION]
        y_test = GNT_list[QUESTION]

        if option == 'all':
            font = {'size': 13}
            matplotlib.rc('font', **font)

            tmp_list = []
            for i in range(N_TRIAL):
                tmp_list.append(np.array(pred_result[i]).reshape(EPOCH_SIZE, -1))
            tmp_list = np.array(tmp_list)

            # pca
            pca = PCA(n_components=2)
            pca.fit(tmp_list[0])
            transformed_result = []
            for i in range(N_TRIAL):
                transformed_result.append(pca.transform(tmp_list[i]))
            transformed_result = np.array(transformed_result)

            # plot
            for iteration in range(N_TRIAL):
                for j in range(EPOCH_SIZE):
                    plt.plot(transformed_result[iteration][j, 0], transformed_result[iteration][j, 1], '.',
                             color=(0, j / EPOCH_SIZE, 1), markersize=j / EPOCH_SIZE * 15)

        else:
            losses, acces = [], []
            for i in range(N_TRIAL):
                loss = np.min(histories[QUESTION][i].history['val_loss'])
                acc = np.max(histories[QUESTION][i].history['val_accuracy'])
                losses.append(loss)
                acces.append(acc)
            if metric == 'loss':
                best_history = np.argmin(losses)
            else:
                best_history = np.argmax(acces)

            font = {'size': 13, 'family': "DejaVu"}
            matplotlib.rc('font', **font)

            tmp_list = []
            for i in range(best_history, best_history + 1):
                tmp_list.append(np.array(pred_result[i]).reshape(EPOCH_SIZE, -1))
            tmp_list = np.array(tmp_list)

            # pca
            pca = PCA(n_components=2)
            pca.fit(tmp_list[0])
            transformed_result = []
            for i in range(1):
                transformed_result.append(pca.transform(tmp_list[i]))
            transformed_result = np.array(transformed_result)

            # plot

            for iteration in range(1):
                for j in range(EPOCH_SIZE):
                    plt.plot(transformed_result[iteration][j, 0], transformed_result[iteration][j, 1], '.',
                             color=(0, j / EPOCH_SIZE, 1), markersize=j / EPOCH_SIZE * 15)

        GNT = y_test.reshape(1, -1)
        GNT_tr = pca.transform(GNT)
        plt.plot(GNT_tr[0, 0], GNT_tr[0, 1], 'rx', label='GNT')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('{}({}) \nTrajectory of prediction result'.format(QUESTION,
                                                                    question_list[QUESTION]))
        plt.legend()
        plt.show()


def plot_pred_result_v2(question_list, histories, question_pred_result, GNT_list, option, metric,  EPOCH_SIZE, N_TRIAL):
    from sklearn.decomposition import PCA

    for QUESTION in question_list.keys():

        pred_result = question_pred_result[QUESTION]
        y_test = GNT_list[QUESTION]

        if option == 'all':
            font = {'size': 13}
            matplotlib.rc('font', **font)

            tmp_list = []
            for i in range(N_TRIAL):
                tmp_list.append(np.array(pred_result[i]).reshape(EPOCH_SIZE, -1))
            tmp_list = np.array(tmp_list)

            # pca
            pca = PCA(n_components=2)
            pca.fit(tmp_list[0])
            transformed_result = []
            for i in range(N_TRIAL):
                transformed_result.append(pca.transform(tmp_list[i]))
            transformed_result = np.array(transformed_result)

            # plot
            for iteration in range(N_TRIAL):
                print(iteration)
                for j in range(EPOCH_SIZE):
                    if iteration <N_TRIAL / 2:
                        color = (0, j / EPOCH_SIZE, 1)
                    else:
                        color = (j / EPOCH_SIZE, 1, 0)
                    plt.plot(transformed_result[iteration][j, 0], transformed_result[iteration][j, 1], '.',
                             color=color, markersize=j / EPOCH_SIZE * 15)

        GNT = y_test.reshape(1, -1)
        GNT_tr = pca.transform(GNT)
        plt.plot(GNT_tr[0, 0], GNT_tr[0, 1], 'rx', label='GNT')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('{}({}) \nTrajectory of prediction result'.format(QUESTION,
                                                                    question_list[QUESTION]))
        plt.legend()
        plt.show()

from scipy.interpolate import interp1d
def dim_reduct(data, window = 24 * 7, th = 0):
    '''
    data를 window 단위로 nan이 있는 주를 제외하함
    th: nan threshold
    '''
    cut = (data.shape[0] // window) * window
    data = data[:cut, :]

    data_3d = data.reshape(-1, window, data.shape[1])  # day, hour, home
    data_3d = data_3d.transpose(2, 0, 1) # home, day, hour
    # nan_day = np.any(pd.isnull(data_3d), axis=1)
    # data_3d = data_3d[:,~nan_day, :]  # nan이 없는 집 삭제
    
    data_list = []
    data_id_list = []
    for i in range(data_3d.shape[0]):
        data_2d = data_3d[i,:,:].copy()
        # nan_idx = np.any(pd.isnull(data_2d), axis=1)

        n_nan = np.sum(pd.isnull(data_2d), axis=1)
        valid_idx = n_nan <= th
        data_2d = data_2d[valid_idx,:]
        
        # 1d interpolation
        data_df = pd.DataFrame(data_2d.T)
        data_df = data_df.interpolate(method='polynomial', order=1)

        # drop nan again
        data_2d = data_df.values.T
        nan_idx = np.any(pd.isnull(data_2d), axis=1)
        data_2d = data_2d[~nan_idx,:]

        if data_2d.size == 0:
            continue
        data_list.append(np.mean(data_2d, axis=0))
        # data_id_list = data_id_list + [i] * data_2d.shape[0]
        data_id_list.append(i)
    # data_list = np.concatenate(data_list, axis=0)
    data_list = np.array(data_list)
    data_id_list = np.array(data_id_list)
    return data_list, data_id_list


# nanmean으로 대표부하 선정
def dim_reduct_v2(data, window = 24 * 7, th = 0):
    '''
    data를 window 단위로 nan이 있는 주를 제정하함
    th: nan threshold
    '''
    cut = (data.shape[0] // window) * window
    data = data[:cut, :]

    data_3d = data.reshape(-1, window, data.shape[1])  # day, hour, home
    data_3d = data_3d.transpose(2, 0, 1) # home, day, hour
    # nan_day = np.any(pd.isnull(data_3d), axis=1)
    # data_3d = data_3d[:,~nan_day, :]  # nan이 없는 집 삭제
    
    data_list = []
    data_id_list = []
    for i in range(data_3d.shape[0]):
        data_2d = data_3d[i,:,:].copy()
        # nan_idx = np.any(pd.isnull(data_2d), axis=1)

        data_2d = np.nanmean(data_2d, axis=0)
        if data_2d.size == 0:
            continue
        data_list.append(data_2d)
        # data_id_list = data_id_list + [i] * data_2d.shape[0]
        data_id_list.append(i)
    # data_list = np.concatenate(data_list, axis=0)
    data_list = np.array(data_list)
    data_id_list = np.array(data_id_list)
    return data_list, data_id_list

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
       
        return acc, y_pred

    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        
        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1 = Xt2, X2 = X, gamma=self.gamma)
        
        # New target features
        Xt2_new = K @ A
        
        return Xt2_new
    
    def fit_predict_new(self, Xt, Xs, Ys, Xt2, Yt2):
        '''
        Transfrom Xt and Xs, get Xs_new
        Transform Xt2 with projection matrix created by Xs and Xt, get Xt2_new
        Make predictions on Xt2_new using classifier trained on Xs_new
        :param Xt: ns * n_feature, target feature
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt2: nt * n_feature, new target feature
        :param Yt2: nt * 1, new target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, _ = self.fit(Xs, Xt)
        Xt2_new   = self.fit_new(Xs, Xt, Xt2)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)
        
        return acc, y_pred