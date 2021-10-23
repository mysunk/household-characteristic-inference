import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feature_extraction(profile_allday):
    features = []
    '''
        mean features
    '''
    # 1. mean P, daily, all day : c_total
    feature = np.mean(profile_allday, axis=1)
    features.append(feature)

    # 4. mean P, 6 ~ 22 : c_day
    feature = np.mean(profile_allday[:, 2 * 6:2 * 22], axis=1)
    features.append(feature)

    # 5. mean P, 6 ~ 8.5: c_morning
    feature = np.mean(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1)
    features.append(feature)

    # 6. mean P, 8.5 ~ 12: c_forenoon
    feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1)
    features.append(feature)

    # 7. mean P, 12 ~ 14.5: c_noon
    feature = np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 8. mean P, 14.5 ~ 18: c_afternoon
    feature = np.mean(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1)
    features.append(feature)

    # 9. mean P, 18 ~ 24: c_evening
    feature = np.mean(profile_allday[:, 2 * 18:2 * 24], axis=1)
    features.append(feature)

    # 10. mean P, 00 ~ 6: c_night
    feature = np.mean(profile_allday[:, :2 * 6], axis=1)
    features.append(feature)

    '''
        max features
    '''
    # 1. max P, daily, all day : c_total
    feature = np.max(profile_allday, axis=1)
    features.append(feature)

    # 4. max P, 6 ~ 22 : c_day
    feature = np.max(profile_allday[:, 2 * 6:2 * 22], axis=1)
    features.append(feature)

    # 5. max P, 6 ~ 8.5: c_morning
    feature = np.max(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1)
    features.append(feature)

    # 6. max P, 8.5 ~ 12: c_forenoon
    feature = np.max(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1)
    features.append(feature)

    # 7. max P, 12 ~ 14.5: c_noon
    feature = np.max(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 8. max P, 14.5 ~ 18: c_afternoon
    feature = np.max(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1)
    features.append(feature)

    # 9. max P, 18 ~ 24: c_evening
    feature = np.max(profile_allday[:, 2 * 18:2 * 24], axis=1)
    features.append(feature)

    # 10. max P, 00 ~ 6: c_night
    feature = np.max(profile_allday[:, :2 * 6], axis=1)
    features.append(feature)
    '''
        min
    '''
    # 1. min P, daily, all day : c_total
    feature = np.min(profile_allday, axis=1)
    features.append(feature)

    # 4. min P, 6 ~ 22 : c_day
    feature = np.min(profile_allday[:, 2 * 6:2 * 22], axis=1)
    features.append(feature)

    # 5. min P, 6 ~ 8.5: c_morning
    feature = np.min(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1)
    features.append(feature)

    # 6. min P, 8.5 ~ 12: c_forenoon
    feature = np.min(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1)
    features.append(feature)

    # 7. min P, 12 ~ 14.5: c_noon
    feature = np.min(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 8. min P, 14.5 ~ 18: c_afternoon
    feature = np.min(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1)
    features.append(feature)

    # 9. min P, 18 ~ 24: c_evening
    feature = np.min(profile_allday[:, 2 * 18:2 * 24], axis=1)
    features.append(feature)

    # 10. min P, 00 ~ 6: c_night
    feature = np.min(profile_allday[:, :2 * 6], axis=1)
    features.append(feature)
    '''
        ratios
    '''
    # 1. mean P over max P
    feature = np.mean(profile_allday, axis=1) / np.max(profile_allday, axis=1)
    features.append(feature)

    # 2. min P over mean P
    feature = np.min(profile_allday, axis=1) / np.mean(profile_allday, axis=1)
    features.append(feature)

    # 3. c_forenoon / c_noon
    feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1) / \
              np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 4. c_afternoon / c_noon
    feature = np.mean(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1) / \
              np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 5. c_evening / c_noon
    feature = np.mean(profile_allday[:, 2 * 18:2 * 24], axis=1) / \
              np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 6. c_noon / c_total
    feature = np.mean(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1) / \
              np.mean(profile_allday, axis=1)
    features.append(feature)

    # 7. c_night / c_day
    feature = np.mean(profile_allday[:, :2 * 6], axis=1) / \
              np.mean(profile_allday[:, 2 * 6:2 * 22], axis=1)
    features.append(feature)

    '''
        temporal properties
    '''
    # 1. P > 0.5
    feature = (profile_allday > 0.5).mean(axis=1)
    features.append(feature)

    # 1. P > 1
    feature = (profile_allday > 1).mean(axis=1)
    features.append(feature)

    # 1. P > 2
    feature = (profile_allday > 2).mean(axis=1)
    features.append(feature)

    '''
        statistical properties
    '''
    # 1. var P, daily, all day : c_total
    feature = np.var(profile_allday, axis=1)
    features.append(feature)

    # 4. var P, 6 ~ 22 : c_day
    feature = np.var(profile_allday[:, 2 * 6:2 * 22], axis=1)
    features.append(feature)

    # 5. var P, 6 ~ 8.5: c_morning
    feature = np.var(profile_allday[:, 2 * 6:2 * 8 + 1], axis=1)
    features.append(feature)

    # 6. var P, 8.5 ~ 12: c_forenoon
    feature = np.var(profile_allday[:, 2 * 8 + 1:2 * 12], axis=1)
    features.append(feature)

    # 7. var P, 12 ~ 14.5: c_noon
    feature = np.var(profile_allday[:, 2 * 12:2 * 14 + 1], axis=1)
    features.append(feature)

    # 8. var P, 14.5 ~ 18: c_afternoon
    feature = np.var(profile_allday[:, 2 * 14 + 1:2 * 18], axis=1)
    features.append(feature)

    # 9. var P, 18 ~ 24: c_evening
    feature = np.var(profile_allday[:, 2 * 18:2 * 24], axis=1)
    features.append(feature)

    # 10. var P, 00 ~ 6: c_night
    feature = np.var(profile_allday[:, :2 * 6], axis=1)
    features.append(feature)

    return np.array(features)

def feature_extraction_v2(profile_allday):
    features = []
    '''
        mean features
    '''
    # 1. mean P, daily, all day : c_total
    feature = np.mean(profile_allday, axis=1)
    features.append(feature)

    # 4. mean P, 6 ~ 22 : c_day
    feature = np.mean(profile_allday[:, 6:22], axis=1)
    features.append(feature)

    # 5. mean P, 6 ~ 8.5: c_morning
    feature = np.mean(profile_allday[:, 6:8], axis=1)
    features.append(feature)

    # 6. mean P, 8.5 ~ 12: c_forenoon
    feature = np.mean(profile_allday[:, 8:12], axis=1)
    features.append(feature)

    # 7. mean P, 12 ~ 14.5: c_noon
    feature = np.mean(profile_allday[:, 12:14], axis=1)
    features.append(feature)

    # 8. mean P, 14.5 ~ 18: c_afternoon
    feature = np.mean(profile_allday[:, 14:18], axis=1)
    features.append(feature)

    # 9. mean P, 18 ~ 24: c_evening
    feature = np.mean(profile_allday[:, 18:24], axis=1)
    features.append(feature)

    # 10. mean P, 00 ~ 6: c_night
    feature = np.mean(profile_allday[:, :6], axis=1)
    features.append(feature)

    '''
        max features
    '''
    # 1. max P, daily, all day : c_total
    feature = np.max(profile_allday, axis=1)
    features.append(feature)

    # 4. max P, 6 ~ 22 : c_day
    feature = np.max(profile_allday[:,  6: 22], axis=1)
    features.append(feature)

    # 5. max P, 6 ~ 8.5: c_morning
    feature = np.max(profile_allday[:, 6:8], axis=1)
    features.append(feature)

    # 6. max P, 8.5 ~ 12: c_forenoon
    feature = np.max(profile_allday[:, 8 : 12], axis=1)
    features.append(feature)

    # 7. max P, 12 ~ 14.5: c_noon
    feature = np.max(profile_allday[:, 12:14], axis=1)
    features.append(feature)

    # 8. max P, 14.5 ~ 18: c_afternoon
    feature = np.max(profile_allday[:, 14:18], axis=1)
    features.append(feature)

    # 9. max P, 18 ~ 24: c_evening
    feature = np.max(profile_allday[:,  18: 24], axis=1)
    features.append(feature)

    # 10. max P, 00 ~ 6: c_night
    feature = np.max(profile_allday[:, :6], axis=1)
    features.append(feature)
    '''
        min
    '''
    # 1. min P, daily, all day : c_total
    feature = np.min(profile_allday, axis=1)
    features.append(feature)

    # 4. min P, 6 ~ 22 : c_day
    feature = np.min(profile_allday[:, 6:22], axis=1)
    features.append(feature)

    # 5. min P, 6 ~ 8.5: c_morning
    feature = np.min(profile_allday[:, 6: 8], axis=1)
    features.append(feature)

    # 6. min P, 8.5 ~ 12: c_forenoon
    feature = np.min(profile_allday[:,  8: 12], axis=1)
    features.append(feature)

    # 7. min P, 12 ~ 14.5: c_noon
    feature = np.min(profile_allday[:, 12: 14], axis=1)
    features.append(feature)

    # 8. min P, 14.5 ~ 18: c_afternoon
    feature = np.min(profile_allday[:,  14: 18], axis=1)
    features.append(feature)

    # 9. min P, 18 ~ 24: c_evening
    feature = np.min(profile_allday[:, 18:24], axis=1)
    features.append(feature)

    # 10. min P, 00 ~ 6: c_night
    feature = np.min(profile_allday[:, : 6], axis=1)
    features.append(feature)
    '''
        ratios
    '''
    # 1. mean P over max P
    feature = np.mean(profile_allday, axis=1) / np.max(profile_allday, axis=1)
    features.append(feature)

    # 2. min P over mean P
    feature = np.min(profile_allday, axis=1) / np.mean(profile_allday, axis=1)
    features.append(feature)

    # 3. c_forenoon / c_noon
    feature = np.mean(profile_allday[:, 8: 12], axis=1) / \
              np.mean(profile_allday[:,  12:14], axis=1)
    features.append(feature)

    # 4. c_afternoon / c_noon
    feature = np.mean(profile_allday[:, 8: 12], axis=1) / \
              np.mean(profile_allday[:,  12:14], axis=1)
    features.append(feature)

    # 5. c_evening / c_noon
    feature = np.mean(profile_allday[:, 18:24], axis=1) / \
              np.mean(profile_allday[:,  12: 14], axis=1)
    features.append(feature)

    # 6. c_noon / c_total
    feature = np.mean(profile_allday[:,  12: 14], axis=1) / \
              np.mean(profile_allday, axis=1)
    features.append(feature)

    # 7. c_night / c_day
    feature = np.mean(profile_allday[:, :6], axis=1) / \
              np.mean(profile_allday[:,  6:22], axis=1)
    features.append(feature)

    '''
        temporal properties
    '''
    # 1. P > 0.5
    feature = (profile_allday > 0.5).mean(axis=1)
    features.append(feature)

    # 1. P > 1
    feature = (profile_allday > 1).mean(axis=1)
    features.append(feature)

    # 1. P > 2
    feature = (profile_allday > 2).mean(axis=1)
    features.append(feature)

    '''
        statistical properties
    '''
    # 1. var P, daily, all day : c_total
    feature = np.var(profile_allday, axis=1)
    features.append(feature)

    # 4. var P, 6 ~ 22 : c_day
    feature = np.var(profile_allday[:,  6: 22], axis=1)
    features.append(feature)

    # 5. var P, 6 ~ 8.5: c_morning
    feature = np.var(profile_allday[:,  6: 8 ], axis=1)
    features.append(feature)

    # 6. var P, 8.5 ~ 12: c_forenoon
    feature = np.var(profile_allday[:,  8: 12], axis=1)
    features.append(feature)

    # 7. var P, 12 ~ 14.5: c_noon
    feature = np.var(profile_allday[:, 12: 14], axis=1)
    features.append(feature)

    # 8. var P, 14.5 ~ 18: c_afternoon
    feature = np.var(profile_allday[:,  14: 18], axis=1)
    features.append(feature)

    # 9. var P, 18 ~ 24: c_evening
    feature = np.var(profile_allday[:, 18: 24], axis=1)
    features.append(feature)

    # 10. var P, 00 ~ 6: c_night
    feature = np.var(profile_allday[:, : 6], axis=1)
    features.append(feature)

    return np.array(features)

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def evaluate_features(data_2d, label, is_categorical = False):
    mi_result = np.zeros((data_2d.shape[1], ))
    corr_result = np.zeros((data_2d.shape[1], ))
    feature_importance_result = np.zeros((data_2d.shape[1], ))

    for i in range(data_2d.shape[1]):
        data = data_2d[:,i]

        '''
        1. MI
        '''
        val, count = np.unique(label, return_counts=True)
        prob = count / count.sum()

        sig_ = np.std(data)
        H_X = 1 / 2 * np.log2(2 * np.pi * np.e * sig_ ** 2)

        H_con = 0
        for ii, v in enumerate(val):
            sig_ = np.std(data[label == v])
            if sig_ == 0:
                continue
            H_con += (1 / 2 * np.log2(2 * np.pi * np.e * sig_ ** 2)) * prob[ii]
        MI = H_X - H_con

        '''
        2. Corr
        '''
        if is_categorical and list(val)!=[0,1]:
            corr_ = 0
            for ii, v in enumerate(val):
                corr_ += np.abs(np.corrcoef(np.ravel(data), np.ravel(label==v))[0,1]) * prob[ii]
        else:
            # print('Not categorical')
            corr_ = np.abs(np.corrcoef(np.ravel(data), np.ravel(label))[0,1])
        corr_result[i] = corr_
        mi_result[i] = MI

    '''
    3. Feature importance
    '''
    # kf = KFold(n_splits=5, shuffle = True, random_state = 0)
    # result_tmp = np.zeros(label.shape)
    # for train_index, test_index in kf.split(data_2d):
    #     X_train, X_test = data_2d[train_index], data_2d[test_index]
    #     y_train, y_test = label[train_index], label[test_index]

    #     rfc = RandomForestRegressor()
    #     # rfc = SVC(kernel='rbf', degree=3)
    #     rfc.fit(X_train, y_train)
    #     rfc_predict = rfc.predict(X_test)
    #     result_tmp[test_index] = rfc_predict
    #     break
    # feature_importance_result = rfc.feature_importances_

  
    return mi_result, corr_result, feature_importance_result

def transform(df, sampling_interv = 24 * 2 * 7):
    '''
    [input]
    df: dataframe (timeseries, home)
    
    [output]
    data_2d: 2d array
    home_arr: home index array
    '''

    # dataframe => 3d numpy array
    n_d, n_h = df.shape
    n_w = n_d // sampling_interv
    n_d = n_w * sampling_interv
    df_rs = df.iloc[:n_d,:].values.T.reshape(n_h, -1, sampling_interv)

    # 3d numpy array => 2d numpy array
    n, m, l = df_rs.shape
    data_2d = df_rs.reshape(n*m, l)
    home_arr = np.repeat(np.arange(0, n), m)
    invalid_idx = np.any(pd.isnull(data_2d), axis=1)
    data_2d = data_2d[~invalid_idx, :]
    home_arr = home_arr[~invalid_idx]

    # constant load filtering
    invalid_idx = np.nanmin(data_2d, axis=1) == np.nanmax(data_2d, axis=1)
    data_2d = data_2d[~invalid_idx, :]
    home_arr = home_arr[~invalid_idx]

    return data_2d, home_arr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
def DNN_model(params, binary, label, n_feature):

    '''

    Parameters
    ----------
    params: DNN learning parameters
    binary: binary or not
    label: y

    Returns
    -------
    model: compiled DNN model

    '''

    x_input = Input(shape=(n_feature,))
    # x_input = Input(shape=(2, 24, 1))
    x = Dense(64, activation='relu', input_shape=(n_feature,))(x_input)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(64)(x)

    # Add svm layer
    x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)

    model = Model(x_input, x_out)
    optimizer = Adam(params['lr'], epsilon=params['epsilon'])

    # model.compile(optimizer=optimizer, loss='squared_hinge')
    if binary:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics =['acc'])
    return model

def plot_history(histories, key='loss'):
    plt.figure(figsize=(10,5))

    for name, history in histories:
      val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
      plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')
      
      if key == 'acc':
        idx = np.argmax(history.history['val_'+key])
      else:
        idx = np.argmin(history.history['val_'+key])
      best_tr = history.history[key][idx]
      best_val = history.history['val_'+key][idx]
      
      # print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr,key, best_val))

    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()

def plot_history_v2(histories, save_path):

    fig, ax1 = plt.subplots(figsize = (10, 5))
    
    key = 'loss'
    for name, history in histories:
        plt.title(name)
        val = ax1.plot(history.epoch, history.history['val_'+key],
                    '--', label='val_loss', color = 'tab:blue')
        ax1.plot(history.epoch, history.history[key], \
                label='train_loss', color = 'tab:blue')
        
        idx = np.argmin(history.history['val_'+key])
        best_tr = history.history[key][idx]
        best_val = history.history['val_'+key][idx]
    ax1.set_ylabel('Cross entropy loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axvline(x=idx, color = 'r')

    key = 'acc'
    ax2 = ax1.twinx()
    for name, history in histories:
        val = ax2.plot(history.epoch, history.history['val_'+key],
                    '--', label='val_acc', color = 'tab:orange')
        ax2.plot(history.epoch, history.history[key], \
                label= 'train_acc', color = 'tab:orange')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_xlabel('Epoch')

    fig.tight_layout()
    # print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr,key, best_val))
    plt.xlabel('Epochs')
    # plt.legend()
    # plt.xlim([0,max(history.epoch)])
    # plt.savefig(save_path, dpi = 100)
    plt.show()


def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

from sklearn import metrics
def evaluate(y_true, y_pred):
    '''
    return acc, auc, f1 score
    '''
    acc_ = (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).mean()
    fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_true, axis=1), y_pred[:,0], pos_label=0)
    auc_ = metrics.auc(fpr, tpr)
    f1_score_ = metrics.f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='weighted')
    # print('accuracy: {:.3f}, auc: {:.3f}, f1 score: {:.3f}'.format(acc_, auc_, f1_score_))

    return acc_, auc_, f1_score_