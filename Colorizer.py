
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import tensorflow as tf

class Model( object ):

    def __init__(self):

        dropout_rate = 0.3
        self.__DIMEN = 64

        self.__NEURAL_SCHEMA = [
            Conv2D( 32 , input_shape=( self.__DIMEN , self.__DIMEN , 1 ) , kernel_size=( 3 , 3 ) , strides=1,activation=relu),
            Dropout( dropout_rate ) ,
            Conv2D( 64, kernel_size=(3, 3), strides=1, activation=relu),
            Dropout(dropout_rate),
            Conv2D( 128, kernel_size=(3, 3), strides=1, activation=relu) ,
            Dropout(dropout_rate),
            Conv2D( 256, kernel_size=(3, 3), strides=1, activation=relu),
            Dropout(dropout_rate),
            Conv2DTranspose( 128, kernel_size=(3, 3), strides=1, activation=relu),
            Dropout(dropout_rate),
            Conv2DTranspose( 64, kernel_size=(3, 3), strides=1, activation=relu),
            Dropout(dropout_rate),
            Conv2DTranspose( 32, kernel_size=(3, 3), strides=1, activation=relu),
            Dropout(dropout_rate),
            Conv2DTranspose( 3, kernel_size=(3, 3), strides=1, activation=relu ),
        ]

        self.__model = tf.keras.Sequential(self.__NEURAL_SCHEMA)

        self.__model.compile(
            optimizer=optimizers.Adam(0.0001),
            loss=losses.mean_squared_error,
            metrics=['mae'],
        )

    def fit(self, X, Y, number_of_epochs):
        self.__model.fit(X, Y, batch_size=3 , epochs=number_of_epochs)
        self.__model.summary()

    def evaluate(self, test_X, test_Y):
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions

    def save_model(self, file_path):
        self.__model.save(file_path)

    def load_model(self, file_path):
        self.__model = models.load_model(file_path)


