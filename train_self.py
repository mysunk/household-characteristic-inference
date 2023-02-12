"""
Train and evaluate self training
"""

from utils.metrics import evaluate
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import argparse

from data_load import HouseholdEnergyDataset
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
es = EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )

dataset = HouseholdEnergyDataset(data_dir = f'data/prepared/{args.dataset}',
                                label_option = args.label_option, 
                                sampling_rate = 2)

data = dataset.data
label = to_categorical(dataset.label, dtype=int)

# get train, val, test data
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = args.val_split, random_state = 0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = args.val_split, random_state = 0)

# train
model = DNN(params, binary = True)
history = model.fit(x_train, y_train, epochs=params['epoch'], \
        verbose=0, validation_data=(x_val, y_val),batch_size=params['batch_size'], callbacks = [es])

# predict
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

# evaluate
train_acc, train_auc, train_f1 = evaluate(y_train, y_train_pred)
val_acc, val_auc, val_f1 = evaluate(y_val, y_val_pred)
test_acc, test_auc, test_f1 = evaluate(y_test, y_test_pred)
