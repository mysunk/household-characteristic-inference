from module.util import *
import tensorflow as tf

import matplotlib
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

## load survey
# survey = pd.read_csv('data/survey_processed_1230.csv')
# survey = pd.read_csv('data/survey_processed_0216.csv')
survey = pd.read_csv('data/survey_processed_0222.csv')

survey['ID'] = survey['ID'].astype(str)
survey.set_index('ID', inplace=True)

## load smart meter data
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
power_df = power_df.iloc[:2*24*28,:] # 4주
power_df['time'] = pd.to_datetime(power_df['time'])
weekday_flag = np.reshape(power_df['time'].dt.weekday.values, (-1, 48))[:, 0]
power_df.set_index('time', inplace=True)


question_list = {
    'Q1': 'Age of building',
    'Q2': 'Retired',
    'Q3': 'Have children',
    'Q4': 'Age of building',
    'Q5': 'Floor area',
    'Q6': 'Buld proportion',
    'Q7': 'House type',
    'Q8': 'Is single',
    'Q9': 'Is family',
    'Q10': 'Cooking facility type',
    'Q11': '# of bedrooms',
    'Q12': '# of appliances',
    'Q13': '# of residents',
    'Q14': 'Social class',
    'Q15': 'House occupancy'
}
#%% valid consumption data flitering
# valid day가 n일 이상 있을 시 dict에 추가
power_dict = dict()
power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48*7))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)
a = []
for i, key in enumerate(power_df.columns):
    if non_nan_idx.sum(axis=1)[i] >=2: # 2주 이상 있음..
        power_dict[key] = []
        # all day
        power_dict[key].append(power_rs[i,non_nan_idx[i],:])

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
            data_tmp = np.mean(power_dict[id][0], axis=0).reshape(1,-1)

            # 48 point를 24 point로 down sampling # FIXME
            data_tmp_down = []
            for i in range(0, 48*7, 2):
                 data_tmp_down.append(np.nanmax(data_tmp[:,i:i+2], axis=1).reshape(-1,1))
            data_tmp_down = np.concatenate(data_tmp_down, axis=1)
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
            non_nan_idx = np.all(~pd.isnull(power_dict[id][0]), axis= 1)
            data_tmp = power_dict[id][0][non_nan_idx,:]
            data_tmp = data_tmp[:2,:] # 2주만 가져옴
            data.append(data_tmp)
            label.append(np.repeat(label_raw.loc[id], 14))
    data = np.concatenate(data, axis=0)
    label = np.ravel(np.array(label))
    return data, label

#%% 데이터 저장
from tqdm import tqdm

data_dict, label_dict, data_ref_dict, label_ref_dict = dict(), dict(), dict(), dict()
for i in tqdm(range(1,16)):
    data_ref, label_ref = matching_ids('Q'+str(i))
    data, label = matching_ids_with_aug('Q' + str(i))

    label_ref_dict['Q' + str(i)] = label_ref
    data_ref_dict['Q' + str(i)] = data_ref

    data_dict['Q'+str(i)] = data
    label_dict['Q'+str(i)] = label


#%% 학습
from hyperopt import hp, tpe, fmin, Trials
from functools import partial
# from util import train

# result dict
result_dict = dict()
CNN_result_dict = dict()
CNN_param_dict = dict()

tuning_algo = tpe.suggest
space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(0.3)),
    'epoch': 1000,
    'batch_size': hp.quniform('batch_size', 32, 32*20, 32),
    'lambda': hp.loguniform('lambda', np.log(1e-5), np.log(1)),
    'epsilon': hp.loguniform('epsilon', np.log(1e-5), np.log(1)),
    'C': hp.uniform('C', 0, 1)
}

# sample
params = {
    'lr': 0.1,
    'epoch': 1000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
N_ITER = 500
for classifier in ['svm', 'softmax']:
    for question_number in tqdm(range(1, 16)):
        print(f'===={question_number}====')

        # load data corresponding to questtion
        data, label_ref = data_ref_dict['Q' + str(question_number)], label_ref_dict['Q' + str(question_number)]

        # bayes opt
        bayes_trials = Trials()
        objective = partial(train, classifier=classifier, data=data, label_ref=label_ref, i=question_number, random_state=1)
        result = fmin(fn=objective, space=space, algo=tuning_algo, max_evals=N_ITER, trials=bayes_trials)

        # save the result
        trials = sorted(bayes_trials.results, key=lambda k: k['loss'])
        CNN_result_dict[question_number] = trials[0]['loss']
        CNN_param_dict[question_number] = trials[0]['params']

        # search한 parameter로 학습
        train(params = CNN_param_dict[question_number], classifier=classifier, data=data, label_ref=label_ref, i=question_number, save_model=True)
        del objective, result, trials, bayes_trials

    result_dict[classifier + '_result'] = CNN_result_dict
    result_dict[classifier + '_param'] = CNN_param_dict

from module.util import save_obj
save_obj(result_dict, 'param_and_result_0223_2')

#%% CER 데이터 학습 -- rf
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
#
# result_rf = []
# for i in tqdm(range(1, 16)):
#     data, label = data_ref_dict['Q'+str(i)], label_ref_dict['Q'+str(i)]
#
#     X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
#
#     model = RandomForestClassifier(n_estimators=100, random_state=0, max_features=5, verbose=True,
#                                    n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     result = (y_pred == y_test).mean()
#
#     result_rf.append(result)
#
# print(result_rf)
# result_rf[7]

#%% AE
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model


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

power_rs = np.reshape(power_df.values.T, (power_df.shape[1], -1, 48*7))
non_nan_idx = np.all(~pd.isnull(power_rs), axis= 2)
power_rs = power_rs[non_nan_idx]
data_source = power_rs.copy()

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
train = data_source[:10000,:].copy()
train = train.reshape(-1, 7, 24, 1)
history_2 = autoencoder_2.fit(train, train, epochs=params['epoch'], verbose=1,
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

plot_history([('with 1000 households', history_2)])
plot_history([('with 10000 households', history)])

# train result
plt.plot(train[0:1].reshape(-1), label='original')
plt.plot(pred_2.reshape(-1), label = 'with 1000 households')
plt.plot(pred.reshape(-1),':', label ='with 10000 households')
plt.legend()
plt.show()


q_results = dict()
question_number = 4
results = dict()
for option in [1, 2, 3]:
    result = []
    for iter in range(100):
        print(f'===={question_number}====')
        # load data corresponding to questtion
        data, label_ref = data_ref_dict['Q' + str(question_number)], label_ref_dict['Q' + str(question_number)]

        label = to_categorical(label_ref.copy(), dtype=int)

        binary = False
        if label.shape[1] == 2:
            binary = True

        label = label.astype(float)

        if option == 1:
            encoder = autoencoder.encoder
            encoder.trainable = True
        elif option == 2:
            encoder = autoencoder_2.encoder
            encoder.trainable = True
        elif option == 3:
            encoder = Autoencoder()

        additional_layer = Sequential([
            Flatten(),
            Dense(32, activation='relu'),
            ])
        prediction_layer = Dense(label.shape[1], input_shape=(16,),
                                  activation='softmax', use_bias=True)

        model = Sequential([encoder, additional_layer, prediction_layer])

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

        model.fit(X_train, y_train, epochs=1000, verbose=0, callbacks=[es], validation_data=(X_test, y_test),
                          batch_size=params['batch_size'])

        model.trainable = False
        y_pred = model.predict(X_test)
        val = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()
        # print(result)
        result.append(val)
    results[option] = result
q_results[question_number] = results

#%% result
import seaborn as sns
for i in range(1, 2):
    results = q_results[i]
    sns.boxplot(data = [results[1], results[2], results[3]])
    plt.ylabel('Accuracy')
    plt.xticks(range(3), ['10000','1000','w/o transfer'])
    # plt.title('Question {}: {}'.format(i, [float(str(np.max(results[j]))[:4]) for j in range(1,4)]))
    plt.title('Question {}'.format(i))
    # plt.tight_layout()
    plt.show()
    break

#%% result
import seaborn as sns
for i in range(7, 8):
    results = q_results[i]
    sns.boxplot(data = [results[2]])
    plt.ylabel('Accuracy')
    plt.xticks(range(1), ['CNN model'])
    plt.title('Question {}'.format(i))
    # plt.tight_layout()
    plt.show()
    break

#%% 간단한 학습 후에 성능 평가
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model


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

pred_result = dict()

for question_number in tqdm(range(13, 14)):
    for iters in range(10):
        print(f'===={question_number}====')

        # load data corresponding to questtion
        data, label_ref = data_ref_dict['Q' + str(question_number)], label_ref_dict['Q' + str(question_number)]
        binary = False
        label = to_categorical(label_ref.copy(), dtype=int)
        if label.shape[1] == 2:
            binary = True

        label = label.astype(float)
        encoder = Autoencoder()
        additional_layer = Sequential([
                    Flatten(),
                    Dense(32, activation='relu'),
                    ])
        prediction_layer = Dense(label.shape[1], input_shape=(16,),
                                  activation='softmax', use_bias=True)

        model = Sequential([encoder, additional_layer, prediction_layer])

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, stratify=label)
        # reshape
        X_train = X_train.reshape(-1, 7, 24, 1)
        X_test = X_test.reshape(-1, 7, 24, 1)

        # optimizer = Adam(0.01, epsilon=0.1)
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
                return lr * tf.math.exp(-0.01)
        ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
        optimizer = Adam(1e-3)
        if binary:
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[es, ls, PredictionCallback(X_test)], validation_data=(X_test, y_test),
                          batch_size=1024)
        pred_result[iters] = y_preds.copy()

import matplotlib.pyplot as plt
def plot_history(histories, key='loss'):
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
EPOCHS = 100
n_iter = 1
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
plt.title('Trajectory of \n prediction result')
plt.legend()
plt.show()