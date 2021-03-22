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

def downsampling(data_tmp):
    '''
    # 48 point를 24 point로 down sampling # FIXME
    '''
    data_tmp_down = []
    for i in range(0, 48 * 7, 2):
        data_tmp_down.append(np.nanmax(data_tmp[:, i:i + 2], axis=1).reshape(-1, 1))
    data_tmp_down = np.concatenate(data_tmp_down, axis=1)
    return data_tmp_down

def load_CER(option = 'target'):
    ## load smart meter data
    power_df = pd.read_csv('data/power_comb.csv')
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
        survey = pd.read_csv('data/survey_processed_0222.csv')
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
            data, label = matching_ids_with_aug('Q' + str(i))

            label_ref_dict['Q' + str(i)] = label_ref
            data_ref_dict['Q' + str(i)] = data_ref

            data_dict['Q' + str(i)] = data
            label_dict['Q' + str(i)] = label

        # return data_dict, label_dict
        return data_ref_dict, label_ref_dict

from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input, UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow as tf


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test):
        self.X_test = X_test
        self.y_preds = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_test)
        self.y_preds.append(y_pred)


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # model
        self.encoder = tf.keras.Sequential([
            Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(6, 22, 8)),
            MaxPool2D(pool_size=(2, 2), input_shape=(4, 20, 16))
        ])
        self.decoder = tf.keras.Sequential([
            UpSampling2D((2, 2)),
            Conv2DTranspose(16, (3, 3), activation='relu'),
            Conv2DTranspose(8, (2, 3), activation='relu'),
            Conv2D(1, (1, 1), activation='relu')
        ])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNN(Model):
    def __init__(self, X_train, y_train):
        super(CNN, self).__init__()

        self.X_train = X_train
        self.y_train = y_train

        self.layer_1 = tf.keras.Sequential([
            Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(6, 22, 8)),
            MaxPool2D(pool_size=(2, 2), input_shape=(4, 20, 16))
        ])
        self.layer_2 = Sequential([
            Flatten(),
            Dense(32, activation='relu'),
        ])
        self.prediction_layer = Dense(y_train.shape[1], input_shape=(16,),
                                 activation='softmax', use_bias=True)


    def call(self, x):
        encoded = self.layer_1(x)
        x = self.layer_2(encoded)
        x_out = self.prediction_layer(x)
        return x_out

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