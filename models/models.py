""" Define deep learning models
"""

from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, UpSampling2D, Conv2DTranspose, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()

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
    def __init__(self):
        super(CNN, self).__init__()

        self.layer_1 = tf.keras.Sequential([
            Conv2D(8, kernel_size=(2, 3), activation='relu', input_shape=(7, 24, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(6, 22, 8)),
            MaxPool2D(pool_size=(2, 2), input_shape=(4, 20, 16))
        ])
        self.layer_2 = Sequential([
            Flatten(),
            Dense(32, activation='relu'),
        ])
        self.prediction_layer = Dense(1, input_shape=(16,),
                                 activation='softmax', use_bias=True)

    def call(self, x):
        encoded = self.layer_1(x)
        x = self.layer_2(encoded)
        x_out = self.prediction_layer(x)
        return x_out


class DNN(Model):
    def __init__(self, params, binary, n_features = 48):
        super(DNN, self).__init__()

        self.params = params
        self.binary = binary

        self.x_input = Input(shape=(n_features,))

        self.dense_1 = Dense(64, activation='relu', input_shape=self.x_input.shape)
        self.dense_2 = Dense(128, activation='relu')
        self.dense_3 = Dense(128, activation='relu')
        self.dense_4 = Dense(64)

        self.out = Dense(1, activation='softmax', use_bias=True)

    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)

        x_out = self.out(x)
        return x_out

    def build(self):
        super(DNN, self).build(self.x_input.shape)
        optimizer = Adam(self.params['lr'], epsilon=self.params['epsilon'])

        if self.binary:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        self.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
