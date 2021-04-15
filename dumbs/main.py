from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
from module.util import *
font = {'size': 16}
matplotlib.rc('font', **font)

## 데이터 로드 및 전처리
power_df = pd.read_csv('../data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
# power_df = power_df.iloc[:2*24*28,:] # 4주
info = load_info(path='data/survey_for_number_of_residents.csv')
datetime = power_df['time']
datetime = pd.to_datetime(datetime)
power_df.set_index('time', inplace=True)
info[info>=7] = 7
label = info['count'].values
label = label.astype(int)

# 대표 프로파일 추출
home_n = power_df.shape[1]
power_arr = power_df.values
power_arr_rs = np.reshape(power_arr.T, (home_n,-1,48)) # home, day, hours
power_arr_rs = np.nanmean(power_arr_rs, axis=1)
power_arr = power_arr_rs
del power_arr_rs

### make data
# 24시간 대표 프로파일
data = power_arr.copy()
# mean
data = np.append(data, np.nanmean(power_arr, axis=1).reshape(-1,1), axis=1)
# median
data = np.append(data, np.nanmedian(power_arr, axis=1).reshape(-1,1), axis=1)
# max
data = np.append(data, np.nanmax(power_arr, axis=1).reshape(-1,1), axis=1)
# std
data = np.append(data, np.nanstd(power_arr, axis=1).reshape(-1,1), axis=1)
# 저녁시간대의 mean
data = np.append(power_arr, np.nanmean(power_arr[:,31:44], axis=1).reshape(-1,1), axis=1)
# proportion
type = 1 # 1: 전체 2: 주말 3: 주중
power_arr =power_df.values.copy()
datetime = pd.to_datetime(datetime)
weekday_flag = np.reshape(datetime.dt.weekday.values, (-1,48))[:,0]
if type == 1:
    weekday_flag[:] = True
elif type == 2:
    weekday_flag = [w >= 5 for w in weekday_flag]
elif type == 3:
    weekday_flag = [w < 5 for w in weekday_flag]
weekday_flag = np.array(weekday_flag).astype(bool)
power_arr_rs = np.reshape(power_arr.T, (power_df.shape[1],-1,48)) # home, day, hours
# weekday_flag = weekday_flag[:7]
power_arr_rs = power_arr_rs[:,weekday_flag,:]
power_arr = np.reshape(power_arr_rs, (power_df.shape[1], -1))
power_arr = power_arr.T
data[pd.isna(data)] = 0
# 그냥 많이 추가해보자..
for upper_th in np.arange(0.1, 2, 0.1):
    num_eff_points = (~pd.isnull(power_arr)).sum(axis=0)
    proportion = (power_arr > upper_th ).sum(axis=0) / num_eff_points
    data = np.append(data, proportion.reshape(-1,1), axis=1)
# features = list(range(48)) + ['mean','med','max','std','mean of evening','proportion']
### make label
label = info['count'].values.copy()
label = label.astype(int)
label = (label >= 3).astype(int)

#%% Feature 분석
results = np.zeros((data.shape[1], 2))
for i in range(data.shape[1]):
    results[i, :] = calc_MI_corr(data[:,i], label)

fig, ax1 = plt.subplots(figsize=(6,3))
# plt.xlabel('Lower bound')
plt.xticks(list(range(len(features)))[::4]
           , features[::4], rotation=30)

ax1.plot(results[:,0], label='MI',color='tab:blue')
ax1.set_ylabel('MI',color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue', )
ax2 = ax1.twinx()
ax2.plot(results[:,1], label='Corr',color='tab:orange')
ax2.set_ylabel('Correlation',color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
# plt.legend()
plt.show()


#%% rf model
from tqdm import tqdm
y_pred = np.zeros(label.shape)
y_true = np.zeros(label.shape)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in tqdm(skf.split(data, label)):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    model = RandomForestRegressor(n_estimators=81, random_state=0, max_features=5)
    model.fit(X_train, y_train)
    y_pred[test_index] = model.predict(X_test)
    y_true[test_index] = y_test

# plot result
y_pred = (y_pred > 0.5).astype(int)
# print(mean_absolute_error(y_true, y_pred))
print((y_true == y_pred).mean())
idx = np.argsort(y_true)
plt.plot(y_true[idx], label='True')
plt.plot(y_pred[idx], label='Predicted')
plt.ylabel('# of residents')
plt.show()

# plot importance score
plt.figure(figsize=(6,3))
plt.plot(model.feature_importances_)
plt.show()


#%% DNN model
import tensorflow as tf
y_pred = np.zeros(label.shape)
y_true = np.zeros(label.shape)
skf = StratifiedKFold(n_splits=5, shuffle=True)

def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

for train_index, test_index in skf.split(data, label):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='linear', input_shape=(data.shape[1],)),
        tf.keras.layers.Dense(48, activation='linear'),
        tf.keras.layers.Dense(24, activation='linear'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(0.0001)

    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    history = model.fit(X_train, y_train, epochs = 300, verbose=100, batch_size=256, validation_data=(X_test, y_test))

    tf.keras.backend.clear_session()
    break

def plot_history(histories, key='mae'):
    plt.figure()

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

    # plt.xlim([0, max(history.epoch)])
    # plt.ylim([0.8, 1])
    plt.show()

plot_history([('DNN model', history)])

#%%
plt.plot(model.get_weights()[0])
plt.show()


#%% SVR model
from tqdm import tqdm
from sklearn.svm import SVR
y_pred = np.zeros(label.shape)
y_true = np.zeros(label.shape)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in tqdm(skf.split(data, label)):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    model = SVR(kernel='rbf',C=80,gamma=0.001)
    model.fit(X_train, y_train)
    y_pred[test_index] = model.predict(X_test)
    y_true[test_index] = y_test

# plot result
print(mean_absolute_error(y_true, y_pred))
# idx = np.argsort(y_true)
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.ylabel('# of residents')
plt.show()

#%% classification
from tqdm import tqdm
from sklearn.svm import SVR
y_pred = np.zeros(label.shape)
y_true = np.zeros(label.shape, dtype=int)
label = info['count'].values.copy()
# label[label>=4] = 4
label = label.astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in tqdm(skf.split(data, label)):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    model = SVR(kernel='rbf',C=80,gamma=0.001)
    model.fit(X_train, y_train)
    y_pred[test_index] = model.predict(X_test)
    y_true[test_index] = y_test

# plot result
print(mean_absolute_error(y_true, y_pred))
idx = np.argsort(y_true)
plt.plot(y_true[idx], label='True')
plt.plot(y_pred[idx],'--', label='Predicted')
plt.ylabel('# of residents')
plt.xlabel('Household index')
plt.legend()
plt.show()

#%% 결과 분석
for v in np.unique(y_true):
    idx = y_true == v
    val = y_pred[idx]
    plt.hist(val, label=v, density=True, alpha=0.8, bins=15)
plt.xlabel('# of residents')
plt.ylabel('Density')
plt.title('P(predicted # of residents|# of residents)')
plt.legend()
plt.show()

#%% accuracy로 바꾸기
from sklearn.metrics import confusion_matrix
import seaborn as sns
font = {'size': 14}
matplotlib.rc('font', **font)

y_pred = np.round(y_pred)
y_pred[y_pred<=0] = 1
y_pred[y_pred>=7] = 7
cm = confusion_matrix(y_true, y_pred, labels=range(1,8))

sns.heatmap(cm/np.sum(cm), annot=True,
            fmt='.1%', cmap='Blues',yticklabels=range(1,8), xticklabels=range(1,8))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


#%%
from pycm import *
font = {'size': 13}
matplotlib.rc('font', **font)

y_pred = y_pred.astype(int)
cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred) # Create CM From Data
print(cm)

df_result = pd.DataFrame.from_dict(cm.normalized_matrix).T
sns.heatmap(df_result, annot=True,fmt='.1%', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()