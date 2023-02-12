"""
By MRMR algorithm, select features and determine optimal K (e. g. number of features)
"""

from mrmr import mrmr_classif
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
import argparse

from data_load import HouseholdEnergyDataset
from utils.metrics import evaluate
from models.models import DNN

parser = argparse.ArgumentParser(description='Self training')
parser.add_argument('dataset', type=str, default='SAVE', help='SAVE or CER')
parser.add_argument('label_option', type=int, default=1, help='select label')
parser.add_argument('val_split', type=float, default=0.25, help='validattion split')
args = parser.parse_args()

params = {
    'lr':0.001,
    'epoch': 10000,
    'batch_size': 128,
    'lambda': 0.01,
    'epsilon': 0.01,
}

dataset = HouseholdEnergyDataset(data_dir = f'data/prepared/{args.dataset}',
                                label_option = args.label_option, 
                                sampling_rate = 2)

data = dataset.data
label = to_categorical(dataset.label, dtype=int)
label = label.astype(float)

# get mrmr features
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.8, random_state = 0, stratify = label)
x_train = pd.DataFrame(x_train)
y_train = pd.Series(y_train)
selected_features = mrmr_classif(x_train, y_train, K = 20)

# find optimal K
VAL_SPLIT = 0.25
result_dict = dict()
for i in range(20):
    
    features = selected_features[i]
    data = dataset.data[:,features]
    label = to_categorical(dataset.label, dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = args.val_split, random_state = 0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = args.val_split, random_state = 0)

    if label.shape[1] == 2:
        binary = True
    else:
        binary = False
    model = DNN(params, binary, n_features = len(features))

    history = model.fit(x_train, y_train, epochs=params['epoch'], \
            verbose=0, validation_data=(x_val, y_val),batch_size=params['batch_size'], callbacks = [es])
    
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
    val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
    test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)
