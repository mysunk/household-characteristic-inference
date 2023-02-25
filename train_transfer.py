"""
Do transfer learning
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from data_load import HouseholdEnergyDataset
from models.models import DNN
from select_instances import SelectInstances
from utils.metrics import evaluate

parser = argparse.ArgumentParser(description='Self training')
parser.add_argument('source_dataset', type=str, default='SAVE')
parser.add_argument('target_dataset', type=str, default='CER')
parser.add_argument('label_option', type=int, default=1, help='select label')
parser.add_argument('val_split', type=float, default=0.25, help='validattion split')
args = parser.parse_args()

source_dataset = HouseholdEnergyDataset(data_dir = f'data/prepared/{args.source_dataset}',
                                label_option = args.label_option, 
                                sampling_rate = 2)
target_dataset = HouseholdEnergyDataset(data_dir = f'data/prepared/{args.target_dataset}',
                                label_option = args.label_option)

mrmr_feature = np.array([1, 32])

# instance selectio
obj = SelectInstances()
valid_idx = obj.get_valid_instance_index(source_dataset = source_dataset, target_dataset = target_dataset)

# train source dataset
data = source_dataset.data[valid_idx,:]
label = to_categorical(source_dataset.label, dtype=int)

# select feature from mrmr result
data = data[:,mrmr_feature]

# get train, val, test data
x_train, x_val, y_train, y_val = train_test_split(data, label, test_size = args.val_split, random_state = 0)

# train
params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}
es = EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )
model = DNN(params, binary = True)
history = model.fit(x_train, y_train, epochs=params['epoch'], \
        verbose=0, validation_data=(x_val, y_val),batch_size=params['batch_size'], callbacks = [es])

# Add layer
base_model = model
model = Sequential()
for layer in base_model.layers[:-1]: # go through until last layer
    model.add(layer)

time_ = np.array(valid_idx)
valid_feature = np.zeros((48), dtype = bool)
valid_feature[time_] = True

inputs = tf.keras.Input(shape=(valid_feature.sum(),))
x = model(inputs, training=False)
# x = tf.keras.layers.Dense(32, activation='relu')(x) # layer 하나 더 쌓음
x_out = Dense(label.shape[1], activation='softmax', use_bias=True)(x)
model = tf.keras.Model(inputs, x_out)

optimizer = Adam(params['lr'], epsilon=params['epsilon'])
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['acc'])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['acc'])
es = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=0,
                    mode='min',
                    restore_best_weights=True
                )

history = model.fit(x_train, y_train, epochs=params['epoch'], verbose=0, callbacks=[es], validation_data=(X_val, y_val),batch_size=params['batch_size'])


# predict
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

# evaluate
train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)
