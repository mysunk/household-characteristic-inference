
# %%
VAL_SPLIT = 0.25
TEST_SPLIT = 0.2

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}

for data_name in ['CER', 'SAVE']:
    for K in range(1, 48):
        if data_name == 'CER':
            tgt_data_name = 'SAVE'
        else:
            tgt_data_name = 'CER'
        
        model_name = data_name + '_src_cv_mrmr_K_'+str(K)
        time_ = np.array(feature_dict[tgt_data_name + '_mrmr_K_47'][:K])
        valid_feature = np.zeros((48), dtype = bool)
        valid_feature[time_] = True

        # model_name = f'{data_name}_src_timeset_{time_idx}'
        # model_name = f'{feature_name}_src_2'
        print(f'DATA:: {data_name}')
            
        label_ref = label_dict[data_name].copy()
        label_ref[label_ref<=2] = 0
        label_ref[label_ref>2] = 1

        data = rep_load_dict[data_name]
        # dataset filtering
        if data_name == 'SAVE':
            data = data[filtered_idx_2,:]
            label_ref = label_ref[filtered_idx_2]
        else:
            data = data[filtered_idx_1,:]
            label_ref = label_ref[filtered_idx_1]
            
        params = make_param_int(params, ['batch_size'])
        label = to_categorical(label_ref.copy(), dtype=int)
        label = label.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SPLIT, random_state = 0, stratify = y_train)

        model = DNN_model(params, True, label, valid_feature.sum())

        es = EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )

        history_self = model.fit(X_train, y_train, epochs=params['epoch'], \
                verbose=0, validation_data=(X_val, y_val),batch_size=params['batch_size'], callbacks = [es])
        model_dict[model_name] = model
        history_dict[model_name] = history_self

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
        val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
        test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)

        plot_history_v2([(model_name, history_self)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')

        # result_df.loc[model_name,:] = [train_acc, val_acc, test_acc,\
        #                                 train_auc, val_auc, test_auc, \
        #                                 train_f1, val_f1, test_f1]

# %%
#  transfer learning with mrmr
import tensorflow as tf
from sklearn.model_selection import LeaveOneOut, KFold
loo = LeaveOneOut()


result_df_2 = pd.DataFrame(columns = ['val_acc', 'test_acc',\
    'val_auc', 'test_auc', \
    'val_f1', 'test_f1'])

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
case = [0.9, 0.95, 0.975, 0.9875, 0.99, 0.995, 0.998, 0.5]
case_idx = 5
TEST_SPLIT = case[case_idx]
VAL_SPLIT = 0.25
for src_data_name, tgt_data_name in zip(['CER', 'SAVE'], ['SAVE', 'CER']):
    for K in range(1, 48):
        time_ = np.array(feature_dict[tgt_data_name + '_mrmr_K_47'][:K])
        valid_feature = np.zeros((48), dtype = bool)
        valid_feature[time_] = True

        TEST_SPLIT = case[case_idx]
        # model_name = f'{feature_name}_case_{case_idx}'
        model_name = src_data_name + '_src_cv_mrmr_K_'+str(K)

        print(f'DATA:: {tgt_data_name}')
            
        label_ref = label_dict[tgt_data_name].copy()
        label_ref[label_ref<=2] = 0
        label_ref[label_ref>2] = 1

        data = rep_load_dict[tgt_data_name]
        params = make_param_int(params, ['batch_size'])
        label = to_categorical(label_ref.copy(), dtype=int)
        label = label.astype(float)

        X_train_raw, X_test, y_train_raw, y_test = train_test_split(data[:,valid_feature], label, test_size = TEST_SPLIT, random_state = 0, stratify = label)
        
        y_pred_tr = np.zeros(y_train_raw.shape)
        y_pred_te_list = []

        kf = KFold(n_splits=5)
        models = []
        for train_index, test_index in kf.split(X_train_raw):
            X_train, X_val = X_train_raw[train_index], X_train_raw[test_index]
            y_train, y_val = y_train_raw[train_index], y_train_raw[test_index]

            # base_model = model_dict[f'{src_data_name}_src_timeset_{time_idx}']
            # base_model = model_dict[f'{feature_name}_src_2']
            base_model = model_dict[model_name]

            model = Sequential()
            for layer in base_model.layers[:-1]: # go through until last layer
                model.add(layer)

            inputs = tf.keras.Input(shape=(valid_feature.sum(),))
            x = model(inputs, training=False)
            # x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
            x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
            model = tf.keras.Model(inputs, x_out)

            optimizer = Adam(params['lr'] * 0.1, epsilon=params['epsilon'])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['acc'])
            es = EarlyStopping(
                                monitor='val_loss',
                                patience=100,
                                verbose=0,
                                mode='min',
                                restore_best_weights=True
                            )
            history_tr_2 = model.fit(X_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_val, y_val),batch_size=params['batch_size'])

            # model_dict[model_name] = model
            # history_dict[model_name] = history_tr_2

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            y_pred_tr[test_index] = y_val_pred
            y_pred_te_list.append(y_test_pred)
            models.append(model)

            # train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
        val_acc, val_auc, val_f1 = evaluate(y_train_raw, y_pred_tr)
        test_acc, test_auc, test_f1 = evaluate(y_test, np.mean(y_pred_te_list, axis=0))

        # plot_history_v2([(model_name, history_tr_2)],save_path = 'household-characteristic-inference/plots/'+model_name+'.png')
        save_name = tgt_data_name + '_cv_mrmr_K_'+str(K)
        result_df_2.loc[save_name,:] = [val_acc, test_acc,\
                                    val_auc, test_auc, \
                                    val_f1, test_f1]
        model_dict[save_name] = model


# %% boxplot
type_ = 'val'
metric = 'auc'
data = 'SAVE'

# plt.figure(figsize = (10, 5))
plt.title(f'{data} dataset {metric.upper()} result for different K')

results, results_2 = [], []
for K in range(1, 48, 1):
    valid_index = [data + f'_cv_mrmr_K_{K}']
    valid_col = [i for i in result_df_2.columns if metric in i]
    valid_col = [i for i in valid_col if type_ in i]
    results.append(result_df_2.loc[valid_index,valid_col].values.reshape(-1))
    valid_col = [i for i in result_df_2.columns if metric in i]
    valid_col = [i for i in valid_col if 'test' in i]
    results_2.append(result_df_2.loc[valid_index,valid_col].values.reshape(-1))

import seaborn as sns
plt.plot(results)
plt.plot(results_2)
plt.xlabel('K')
# plt.xticks(range(46)[::3], range(1, 48, 1)[::3])
plt.ylabel(metric.upper())
plt.show()
