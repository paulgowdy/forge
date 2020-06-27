#from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, Flatten, Dense

import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input,Lambda, Subtract, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop


class ConvolutionalNeuralNetwork:

    def __init__(self, input_shape, action_space):
        '''
        self.model = Sequential()
        self.model.add(Conv2D(4,
                              3,
                              strides=(1, 1),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape))
        self.model.add(Conv2D(4,
                              3,
                              strides=(1, 1),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape))

        self.model.add(Conv2D(64,
                              3,
                              strides=(1, 1),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape))

        self.model.add(Flatten())
        self.model.add(Dense(8, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()
        '''
        #print('Keras funtional API!')
        board_input = Input(shape=(input_shape))
        #input2 = keras.layers.Input(shape=(1,))
        #merged = keras.layers.Concatenate(axis=1)([input1, input2])
        scalar_inputs = Input(shape=(2,))
        scalar_dense1 = Dense(16)(scalar_inputs)
        scalar_dense2 = Dense(8)(scalar_dense1)

        conv1 = Conv2D(64, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(board_input)
        conv2 = Conv2D(64, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(conv1)
        flat = Flatten()(conv2)

        combined = Concatenate(axis=1)([flat, scalar_dense2])

        dense1 = Dense(32)(combined)
        dense2 = Dense(16)(dense1)

        actions = Dense(action_space)(dense2)

        self.model = Model([board_input, scalar_inputs], actions)
        self.model.compile(RMSprop(lr=0.00025,
                          rho=0.95,
                          epsilon=0.01), loss=tf.keras.losses.MeanSquaredError(),
                          metrics=["accuracy"])
        self.model.summary()
