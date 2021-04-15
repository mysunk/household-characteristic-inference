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
import tensorflow as tf


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test):
        self.X_test = X_test
        self.y_preds = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_test)
        self.y_preds.append(y_pred)


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # model
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


class CNN(Model):
    def __init__(self, X_train, y_train):
        super(CNN, self).__init__()

        self.X_train = X_train
        self.y_train = y_train

        self.layer_1 = tf.keras.Sequential([
            Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(6, 22, 8)),
            MaxPool2D(pool_size=(2, 2), input_shape=(4, 20, 16))
        ])
        self.layer_2 = Sequential([
            Flatten(),
            Dense(32, activation='relu'),
        ])
        self.prediction_layer = Dense(y_train.shape[1], input_shape=(16,),
                                 activation='softmax', use_bias=True)


    def call(self, x):
        encoded = self.layer_1(x)
        x = self.layer_2(encoded)
        x_out = self.prediction_layer(x)
        return x_out
