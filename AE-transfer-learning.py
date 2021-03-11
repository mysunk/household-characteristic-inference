import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
font = {'size': 16, 'family':"Malgun godic"}
matplotlib.rc('font', **font)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# -------------------
# 데이터 로드
# -------------------

data_path = 'D:/GITHUB/python_projects/etri-load-preprocessing/'

def load_energy(data_p, start_date, end_date, additional = None):
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


start_date = pd.to_datetime('2018-07-14 00:00:00')
end_date = pd.to_datetime('2018-08-10 23:00:00')

data_source_p = 'D:/ISP/8. 과제/2020 ETRI/data/SG_data_서울_비식별화/1147010100.csv'
data_target_p = 'D:/ISP/8. 과제/2020 ETRI/data/label_data.csv'

# data
data_source = load_energy(data_source_p, start_date, end_date)
data_target = load_energy(data_target_p, start_date, end_date, additional = True)

# label
def load_extra_info(idx):
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
                        'ap_7', 'ap_8', 'ap_9','ap_10', 'ap_11',
                        'm_0', 'm_10', 'm_20', 'm_30', 'm_40', 'm_50', 'm_60', 'm_70',
                        'w_0', 'w_10', 'w_20', 'w_30', 'w_40', 'w_50', 'w_60', 'w_70',
                        'income_type','work_type']
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

extra_info = load_extra_info(idx=data_target.columns).astype(int)

# make 3d data
def dimension_reduction(data):
    data_3d = data.reshape(-1, 24*7, data.shape[1]) # day, hour, home
    data_3d = data_3d.transpose(2,0,1)
    data_2d = np.nanmean(data_3d, axis=1)
    return data_2d

data_source = dimension_reduction(data_source.values)
data_target = dimension_reduction(data_target.values)

# nan이 있는 집 삭제
nan_home = np.any(pd.isnull(data_source), axis=1)
data_source = data_source[~nan_home, :]
nan_home = np.any(pd.isnull(data_target), axis=1)
data_target = data_target[~nan_home, :]
extra_info = extra_info.iloc[:,~nan_home]

# 1. number of residents
label_residents = extra_info.loc['popl_num',:].values.copy()
label_residents[label_residents <=2] = 0
label_residents[label_residents >2] = 1

# 2. number of appliances
label_appliances = extra_info.loc['appl_num',:].values.copy()
label_appliances[label_appliances<=6] = 0
label_appliances[(label_appliances> 6) * (label_appliances <= 8)] = 1
label_appliances[label_appliances > 8] = 2

# 3. have child
label_child = extra_info.loc['child_num',:].values.copy()
label_child[label_child>0] = 1

# 4. have child include teen
label_child_w_teen = extra_info.loc['child_include_teen',:].values.copy()
label_child_w_teen[label_child_w_teen>0] = 1

# 5. single
label_single = (extra_info.loc['adult_num',:].values == 1) * (extra_info.loc['child_include_teen',:].values == 0)
label_single = label_single.astype(int)

# 6. area
label_area = extra_info.loc['area',:].values.copy()
label_area[label_area < 20] = 0
label_area[(label_area >= 20) * (label_area <= 22)] = 1
label_area[label_area > 22] = 2

labels = [label_residents, label_appliances, label_child, label_child_w_teen, label_single, label_area]

question_list = {
    'Q1': '# of residents',
    'Q2': '# of appliances',
    'Q3': 'Have children under 10',
    'Q4': 'Have children under 20',
    'Q5': 'Is single',
    'Q6': 'Floor area'
}

#%%
# -------------------
# AE 모델 학습
# -------------------
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

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
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

from sklearn.model_selection import train_test_split

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=0,
    mode='min',
    restore_best_weights=True
)
params = {
    'epoch': 10000,
    'batch_size': 128,
    'epsilon': 0.1,
}

autoencoder = Autoencoder()
# optimizer = opt = SGD(lr=params['lr'], momentum=0.9, decay=1e-2/params['epoch'])
optimizer = SGD(0.1, momentum=0.9)
autoencoder.compile(optimizer = optimizer, loss='mse')
train = data_source[:1000,:].copy()
train = train.reshape(-1, 7, 24, 1)
history = autoencoder.fit(train, train, epochs=params['epoch'], verbose=1,
          batch_size=params['batch_size'], validation_split=0.2,
                callbacks = [es])
pred = autoencoder.predict(train[0:1])

autoencoder_2 = Autoencoder()
# optimizer = opt = SGD(lr=params['lr'], momentum=0.9, decay=1e-2/params['epoch'])
optimizer = SGD(0.1, momentum=0.9)
autoencoder_2.compile(optimizer = optimizer, loss='mse')
train = data_source[:1000,:].copy()
train = train.reshape(-1, 7, 24, 1)
history_2 = autoencoder_2.fit(train, train, epochs=3, verbose=1,
          batch_size=params['batch_size'], validation_split=0.2,
                callbacks = [es])
pred_2 = autoencoder_2.predict(train[0:1])

# evaluate result
import matplotlib.pyplot as plt
def plot_history(histories, key='loss'):
    plt.figure(figsize=(10, 5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

        idx = np.argmin(history.history['val_' + key])
        best_tr = history.history[key][idx]
        best_val = history.history['val_' + key][idx]

        print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr, key, best_val))

    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()

plot_history([('EPOCH: 10', history_2)])
plot_history([('EPOCH: {}'.format(params['epoch']), history)])

plt.plot(train[0:1].reshape(-1), label='original')
plt.plot(pred_2.reshape(-1), label = '10 EPOCHs')
plt.plot(pred.reshape(-1),':', label ='{} EPOCHs'.format(params['epoch']))
plt.xlabel('Hour')
plt.ylabel('Energy consumption [kW]')
plt.legend()
plt.show()

#%% 예측값 확인
question_pred_result = dict()
GNT_list = dict()
from collections import defaultdict
histories = defaultdict(list)

n_iter = 30
for question_number in range(1, 7):
    pred_result = dict()
    for iters in range(n_iter):
        label_raw = labels[question_number-1].copy()
        label = to_categorical(label_raw, dtype=int)
        data = data_target
        binary = False
        if label.shape[1] == 2:
            binary = True

        label = label.astype(float)
        encoder = Autoencoder()
        additional_layer = Sequential([
                    Flatten(),
                    Dense(32, activation='relu'),
                    ])
        prediction_layer = Dense(label.shape[1], input_shape=(16,) ,
                                  activation='softmax', use_bias=True)

        model = Sequential([encoder, additional_layer, prediction_layer])

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
        # reshape
        X_train = X_train.reshape(-1, 7, 24, 1)
        X_test = X_test.reshape(-1, 7, 24, 1)

        # optimizer = Adam(0.01, epsilon=0.1)
        from collections import defaultdict
        y_preds = []
        y_true = np.ravel(y_test)
        class PredictionCallback(tf.keras.callbacks.Callback):
            def __init__(self, val):
                self.val = val

            def on_epoch_end(self, epoch, logs={}):
                y_pred = self.model.predict(self.val)
                # save the result in each training
                y_preds.append(y_pred)
                # np.save('preds_w_pretraining_3/' + str(epoch), y_pred)


        es = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )
        def scheduler(epoch, lr):
            # print(lr)
            if epoch < 40:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
        optimizer = Adam(1e-4)
        if binary:
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=100, verbose=0,
                            callbacks=[es, ls, PredictionCallback(X_test)], validation_data=(X_test, y_test),
                          batch_size=params['batch_size'])
        histories[question_number].append(history)
        pred_result[iters] = y_preds.copy()
    question_pred_result[question_number] = pred_result
    GNT_list[question_number] = y_test

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

plt.subplot(2,1,1)
idx = np.argmin(history.history['val_loss'])
plt.title('{}({}) loss:{:.2f}, acc:{:.2f} '.format('Q' + str(question_number), question_list['Q' + str(question_number)],history.history['val_loss'][idx], history.history['val_accuracy'][idx]))
plot_history([('', history)], key = 'loss')
plt.subplot(2,1,2)
plot_history([('', history)], key = 'accuracy')
plt.legend()
plt.show()

#%%
question_number = 6
pred_result = question_pred_result[question_number]
y_test = GNT_list[question_number]

EPOCHS = 100
font = {'size': 13, 'family':"Malgun godic"}
matplotlib.rc('font', **font)

tmp_list = []
for i in range(n_iter):
    tmp_list.append(np.array(pred_result[i]).reshape(EPOCHS,-1))
tmp_list = np.array(tmp_list)

# pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(tmp_list[0])
transformed_result = []
for i in range(n_iter):
    transformed_result.append(pca.transform(tmp_list[i]))
transformed_result = np.array(transformed_result)

# plot
GNT = y_test.reshape(1, -1)
GNT_tr = pca.transform(GNT)
for iteration in range(n_iter):
    for j in range(EPOCHS):
        plt.plot(transformed_result[iteration][j,0], transformed_result[iteration][j,1],'.',
                 color = (0,j / EPOCHS,1), markersize = j / 5)
plt.plot(GNT_tr[0,0], GNT_tr[0,1],'rx', label='GNT')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('{}({}) \nTrajectory of prediction result'.format('Q' + str(question_number), question_list['Q' + str(question_number)]))
plt.legend()
plt.show()

#%%
for i in range(10):
    for j in range(100):
        print((np.argmax(pred_result[i][j], axis=1) == 1).mean())

#%% pca coefficient
plt.plot(pca.components_.T)
plt.xlabel('coefficient index')
plt.title('PCA coefficient')
plt.legend(['1st component','2nd component'])
plt.show()

#%% fine tuning
results = dict()
for option in [6]:
    result = []
    for iter in range(10):
        i = 1
        label_raw = labels[i].copy()
        label = to_categorical(label_raw, dtype=int)
        data = data_target
        binary = False
        if label.shape[1] == 2:
            binary = True

        label = label.astype(float)

        if option == 1:
            encoder = autoencoder.encoder
            encoder.trainable = False
        elif option == 2:
            encoder = autoencoder_2.encoder
            encoder.trainable = False
        elif option == 3:
            encoder = Autoencoder()
        elif option == 4:
            encoder = autoencoder.encoder
            encoder.trainable = True
        elif option == 5:
            encoder = autoencoder_2.encoder
            encoder.trainable = True
        elif option == 6:
            encoder = Autoencoder()

        additional_layer = Sequential([
            Flatten(),
            Dense(32, activation='relu'),
            ])
        prediction_layer = Dense(label.shape[1], input_shape=(16,),
                                  activation='softmax', use_bias=True)

        model = Sequential([encoder , additional_layer, prediction_layer])

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
        # reshape
        X_train = X_train.reshape(-1, 7, 24, 1)
        X_test = X_test.reshape(-1, 7, 24, 1)

        # optimizer = Adam(0.01, epsilon=0.1)
        optimizer = SGD(0.01, momentum=0.9)
        if binary:
            model.compile(optimizer=optimizer, loss='binary_crossentropy')
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        model.fit(X_train, y_train, epochs=1000, verbose=1, callbacks=[es], validation_data=(X_test, y_test),
                          batch_size=params['batch_size'])

        model.trainable = False
        y_pred = model.predict(X_test)
        val = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()
        # print(result)
        result.append(val)
    results[option] = result

#%%
import seaborn as sns
sns.boxplot(data = results[6])
plt.ylabel('Accuracy')
# plt.xticks(range(3), ['1000','100','w/o transfer'])
plt.show()

#%%
for i in range(1, 7):
    print(np.max(results[i]))