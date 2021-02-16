import os

os.chdir('../')
from util_etri import *
import matplotlib
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

font = {'size': 15}
matplotlib.rc('font', **font)


#%% Input
season = 0 # 0: winter 1: spring 2: summer 3: winter 4
# : all
wnw = 1 # 1:all 2:w 3:nw

start_dates = pd.to_datetime(np.array(['2018-12-01 00:00:00', '2019-04-01 00:00:00', '2019-07-01 00:00:00', '2019-10-01 00:00:00','2018-12-01 00:00:00']))
end_dates = pd.to_datetime(np.array(['2018-12-31 23:00:00', '2019-04-30 23:00:00', '2019-07-31 23:00:00', '2019-10-31 23:00:00','2019-10-31 23:00:00']))

start_date = pd.to_datetime('2018-07-14 00:00:00')
end_date = pd.to_datetime('2018-08-10 23:00:00')

calendar = load_calendar(start_date, end_date)  # 1-nw, 0-w
energy = load_energy(start_date, end_date, None)
extra_info = load_extra_info(idx=energy.columns).astype(int)
extra_info.loc['appl_num',:] = extra_info.loc['ap_1':'ap_11',:].sum(axis=0)
extra_info.loc['popl_num', :] = extra_info.loc['m_0':'w_70', :].sum(axis=0)
extra_info.loc['adult_num', :] = (extra_info.loc['m_30':'m_70', :].values + extra_info.loc['w_30':'w_70', :].values).sum(axis=0)
extra_info.loc['erderly_num', :] = (extra_info.loc['m_60':'m_70', :].values + extra_info.loc['w_60':'w_70', :].values).sum(axis=0)
extra_info.loc['child_num', :] = extra_info.loc['m_0', :].values + extra_info.loc['w_0', :].values
extra_info.loc['teen_num', :] = extra_info.loc['m_10', :].values + extra_info.loc['w_10', :].values
extra_info.loc['child_include_teen', :] = (extra_info.loc['m_0':'m_10', :].values + extra_info.loc['w_0':'w_10', :].values).sum(axis=0)
extra_info.loc['male_num', :] = extra_info.loc['m_0':'m_70', :].sum(axis=0)
extra_info.loc['female_num', :] = extra_info.loc['w_0':'w_70', :].sum(axis=0)
extra_info.loc['income_solo', :] = (extra_info.loc['income_type', :] == 2).astype(int)
extra_info.loc['income_dual', :] = (extra_info.loc['income_type', :] == 1).astype(int)
extra_info.loc['work_home', :] = (extra_info.loc['work_type', :] == 1).astype(int)
extra_info.loc['work_office', :] = (extra_info.loc['work_type', :] == 2).astype(int)
extra_info.drop(index=extra_info.index[12:28], inplace=True)
extra_info.drop(index=['income_type','work_type'], inplace=True)
extra_info = extra_info.astype(int)

# make 3d data
data_3d = energy.values.reshape(-1, 24*7, energy.shape[1]) # day, hour, home

data_3d = data_3d.transpose(2,0,1)
data_2d = np.nanmean(data_3d, axis=1)

nan_home = np.any(pd.isnull(data_2d), axis=1)
data_2d = data_2d[~nan_home,:]
extra_info = extra_info.iloc[:,~nan_home]

# # nan이 하루라도 있는 날은 다같이 제외
# day = data_3d.shape[0]
# is_nan_day = np.zeros(day).astype(bool)
# for d in range(day):
#     # print(np.sum(np.ravel(np.isnan(data_3d[i,:,:]))))
#     is_nan_day[d] = np.any(np.ravel(np.isnan(data_3d[d,:,:])))
# data_3d = data_3d[~is_nan_day, :, :]
# day, _, home = data_3d.shape

#%%
# data_2d = data_3d.mean(axis=1)
# data_2d = data_2d.T

# 1. number of residents
label_residents = extra_info.loc['popl_num',:].values.copy()
label_residents[label_residents <=2] = 1
label_residents[label_residents >2] = 2

# 2. number of appliances
label_appliances = extra_info.loc['appl_num',:].values.copy()
label_appliances[label_appliances<=6] = 1
label_appliances[(label_appliances> 6) * (label_appliances <= 8)] = 2
label_appliances[label_appliances > 8] = 3

# 3. have child
label_child = extra_info.loc['child_include_teen',:].values.copy()
label_child[label_child>0] = 1

# 4. single
label_single = (extra_info.loc['adult_num',:].values == 1) * (extra_info.loc['child_include_teen',:].values == 0)
label_single = label_single.astype(int)

# 5. area
label_area = extra_info.loc['area',:].values.copy()
label_area[label_area < 20] = 1
label_area[(label_area >= 20) * (label_area <= 22)] = 2
label_area[label_area > 22] = 3

#%% w/o augmented: RF
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

result_rf = []
for label in [label_residents, label_appliances, label_child, label_single, label_area]:
    data = data_2d
    y_pred = np.zeros(label.shape)
    y_true = np.zeros(label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = RandomForestClassifier(n_estimators=100, random_state=0, max_features=5, verbose=True,
                                       n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)
        y_true[test_index] = y_test

    # acc
    print((y_pred == y_true).mean())

    # uniform guess
    v, c = np.unique(label, return_counts=True)
    print(c.max() / c.sum())

    result_rf.append((y_pred == y_true).mean())

#%% 저장할 dictionary 선언
labels = [label_residents, label_appliances, label_child, label_single, label_area]

CNN_result_dict = dict()
CNN_param_dict = dict()

#%% CNN 학습
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from functools import partial
from util import train

space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(0.3)),
    'epoch': 100,
    'batch_size': hp.quniform('batch_size', 32, 32*20, 32),
    'lambda': hp.loguniform('lambda', np.log(1e-5), np.log(1)),
    'epsilon': hp.loguniform('epsilon', np.log(1e-5), np.log(1)),
}

for question_number in range(5):
    bayes_trials = Trials()
    tuning_algo = tpe.suggest
    objective = partial(train, data = data_2d, label_ref = labels[question_number], i = question_number)
    result = fmin(fn=objective, space=space, algo=tuning_algo, max_evals=30, trials=bayes_trials)
    trials = sorted(bayes_trials.results, key=lambda k: k['loss'])
    CNN_param_dict[question_number] = trials[0]['params']
    CNN_result_dict[question_number] = trials[0]['loss']

#%% transfer learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

# question_list = [13, 12, 3, 8, 5]
# models = []
# for q in [3]:
#     models.append(tf.keras.models.load_model(f"models/model_Q_{q}.h5")) # number of residents

result_tr = []
for i, question_number in tqdm(enumerate([13, 12, 3, 8, 5])):
    params = CNN_param_dict[i]
    params['lr'] = params['lr'] / 10
    params['epoch'] = 100

    label = labels[i]
    label_raw = label.copy()
    label = to_categorical(label, dtype=int)
    data = data_2d
    binary = False
    if np.all(np.unique(label_raw) == np.array([0, 1])):
        binary = True
        label = label[:, 1]

    label = label.astype(float)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
    """ 
    CNN based feature selection
    """
    # reshape
    X_train = X_train.reshape(-1, 7, 24, 1)
    X_test = X_test.reshape(-1, 7, 24, 1)

    base_model = tf.keras.models.load_model(f"models/model_Q_{question_number}.h5")
    base_model.trainable = False

    model = Sequential()
    for layer in base_model.layers[:-1]: # go through until last layer
        model.add(layer)

    inputs = tf.keras.Input(shape=(7, 24, 1))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = model(inputs, training=False)

    # layer 하나 더 쌓음
    x = tf.keras.layers.Dense(16, activation='relu')(x)

    if binary:
        x_out = Dense(1, input_shape=(16,),
                      activation='sigmoid', use_bias=True)(x)
    else:
        x_out = Dense(label.shape[1], input_shape=(16,),
                      activation='softmax', use_bias=True)(x)

    model = tf.keras.Model(inputs, x_out)

    # compile 및 학습
    optimizer = Adam(params['lr'], epsilon=params['epsilon'])
    if binary:
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )
    model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_test, y_test),
              batch_size=params['batch_size'])

    if binary:
        y_pred = model.predict(X_test).reshape(-1)
        y_pred = (y_pred > 0.5).astype(int)
        result = (y_pred == y_test).mean()
    else:
        y_pred = model.predict(X_test)
        result = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()

    result_tr.append(result)
print(result_tr)
