import numpy as np
import pandas as pd
# import tensorflow as tf
#
# tf.compat.v1.set_random_seed(42)
np.random.seed(42)

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


def CNN_softmax(params, binary, label):

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

    x_input = Input(shape=(7, 24, 1))
    x = Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1))(x_input)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(32, input_shape=(320,))(x)

    # Add svm layer
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

def svm_loss(layer, C):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        regularization_loss = C * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss + 0.5 * hinge_loss

    return categorical_hinge_loss

import tensorflow as tf
def CNN_svm(params, binary, label):
    x_input = Input(shape=(7, 24, 1))
    x = Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1))(x_input)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(32, input_shape=(320,))(x)

    # Add svm layer
    x_out = Dense(label.shape[1], input_shape=(32,),
       use_bias=False, activation='linear', name='svm')(x)

    model = Model(x_input, x_out)
    optimizer = Adam(params['lr'], epsilon=params['epsilon'])
    # optimizer = tf.keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)
    model.compile(optimizer=optimizer, loss=svm_loss(model.get_layer('svm'), params['C']))

    return model

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split

class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val, save_pred_name):
        self.val = val
        self.save_pred_name = save_pred_name

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.val)
        # save the result in each training
        if self.save_pred_name:
            np.save(self.save_pred_name + '/'+str(epoch), y_pred)

def train(params, data, label_ref, classifier, i = 1, save_model = False, random_state = 0, save_pred_name = None):
    # print(classifier)
    params = make_param_int(params, ['batch_size'])

    label = to_categorical(label_ref.copy(), dtype=int)
    # binary or multi
    if label.shape[1] == 2:
        binary = True
    else:
        binary = False

    label = label.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = random_state, stratify = label)
    """ 
    CNN based feature selection
    """
    # reshape
    X_train = X_train.reshape(-1, 7, 24, 1)
    X_test = X_test.reshape(-1, 7, 24, 1)
    if classifier == 'softmax':
        model = CNN_softmax(params, binary, label)
    elif classifier == 'svm':
        model = CNN_svm(params, binary, label)

    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )

    model.fit(X_train, y_train, epochs=1000, verbose=0, callbacks=[es, PredictionCallback(X_test, save_pred_name)], validation_data=(X_test, y_test),
              batch_size=params['batch_size'])

    if save_model:
        model.save("models/model_Q_" + str(i) + f'_{classifier}.h5')

    y_pred = model.predict(X_test)
    # print(y_pred)
    result = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()

    return {'loss': -result, 'params': params, 'status': STATUS_OK, 'model':model}

import pickle
def save_obj(obj, name):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        # trials = sorted(trials, key=lambda k: k['loss'])
        return trials