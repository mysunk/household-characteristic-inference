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

from scipy import stats
def calc_MI_corr(data, label, is_categorical):
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
    if is_categorical and list(val)!=[0,1]:
        corr_ = 0
        for ii, v in enumerate(val):
            corr_ += np.abs(np.corrcoef(np.ravel(data), np.ravel(label==v))[0,1]) * prob[ii]
    else:
        print('Not categorical')
        corr_ = np.abs(np.corrcoef(np.ravel(data), np.ravel(label))[0,1])
    return [MI, corr_]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
def baseline_model(params, binary, label):

    x_input = Input(shape=(7, 24, 1))
    x = Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1),
           kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
           bias_initializer=initializers.Ones())(x_input)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
           bias_initializer=initializers.Ones())(x)
    x = MaxPool2D()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(32, input_shape=(320,), kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
           bias_initializer=initializers.Ones())(x)

    # Add svm layer
    if binary:
        # x_out = Dense(1, input_shape=(32,), kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
        #    bias_initializer=initializers.Ones(),
        #               activation='linear',use_bias=True, name='svm', kernel_regularizer=regularizers.l2(params['lambda']))(x)
        x_out = Dense(1, input_shape=(32,),
                      activation='sigmoid', use_bias=True)(x)
        # x_out = tf.math.sign(x_out)
    else:
        # x_out = Dense(label.shape[1], input_shape=(32,),kernel_initializer=initializers.RandomNormal(stddev=0.01, mean=0),
        #    bias_initializer=initializers.Ones(),
        #               use_bias=True, activation='linear', name='svm', kernel_regularizer=regularizers.l2(params['lambda']))(x)
        x_out = Dense(label.shape[1], input_shape=(32,),
                      activation='softmax', use_bias=True)(x)

    model = Model(x_input, x_out)
    optimizer = Adam(params['lr'], epsilon=params['epsilon'])
    # model.compile(optimizer=optimizer, loss='squared_hinge')
    if binary:
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split
def train(params, data, label_ref, i = 1, save_model = False, random_state = 0):
    params = make_param_int(params, ['batch_size'])

    label_ref = label_ref.astype(float)
    binary = False
    label = to_categorical(label_ref.copy(), dtype=int)
    # binary or multi
    if np.all(np.unique(label_ref) == np.array([0, 1])):
        binary = True
        label = label[:, 1]

    # label = label.astype(float)
    # for svm
    # label[label == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = random_state, stratify = label)
    """ 
    CNN based feature selection
    """
    # reshape
    X_train = X_train.reshape(-1, 7, 24, 1)
    X_test = X_test.reshape(-1, 7, 24, 1)
    model = baseline_model(params, binary, label)
    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )
    model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_test, y_test),
              batch_size=params['batch_size'])

    if save_model:
        model.save("models/model_Q_" + str(i) + '.h5')

    if binary:
        y_pred = model.predict(X_test).reshape(-1)
        # print(y_pred)
        # y_pred = np.sign(y_pred)
        y_pred = (y_pred > 0.5).astype(int)
        result = (y_pred == y_test).mean()
    else:
        y_pred = model.predict(X_test)
        result = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()
    return {'loss': -result, 'params': params, 'status': STATUS_OK}
