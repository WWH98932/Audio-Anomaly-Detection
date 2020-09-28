#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   1d_AE.py
@Time    :   2020/09/28 11:12:05
@Author  :   Wang Minxuan 
@Version :   1.0.0
@Contact :   mx.wang@cyber-insight.com
@License :   (C)Copyright 2019-2020, CyberInsight
@Desc    :   None
'''

# here put the import lib
import keras.models
from keras.models import Model
from keras.layers import *

def dense_ae(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))
    h = Dense(128)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(inputDim)(h)
    return Model(inputs=inputLayer, outputs=h)

def 1d_cae(inputDim): # inputDim: 128 * 5 = 640
    '''
    define 1d convolutional like autoencoder
    the number of filters can be adjusted, kernel size 3 with dilation rate 2 is similar with kernel size 5
    the shape of layer d1 and d2 is determined by the output shape of layer x6
    '''
    inputLayer = Input(shape=(inputDim, 1))
    x = Conv1D(64, 3, activation='relu', padding='same', dilation_rate=2)(inputLayer)
    x1 = MaxPooling1D(2)(x)
    x2 = Conv1D(32, 3, activation='relu', padding='same', dilation_rate=2)(x1)
    x3 = MaxPooling1D(2)(x2)
    x4 = Conv1D(16, 3, activation='relu', padding='same', dilation_rate=2)(x3)
    x5 = MaxPooling1D(2)(x4)
    x6 = AveragePooling1D()(x5)
    flat = Flatten()(x6)
    encoded = Dense(2)(flat)
    d1 = Dense(640)(encoded)
    d2 = Reshape((40, 16))(d1)
    d3 = Conv1D(16, 1, strides=1, activation='relu', padding='same')(d2)
    d4 = UpSampling1D(2)(d3)
    d5 = Conv1D(32, 1, strides=1, activation='relu', padding='same')(d4)
    d6 = UpSampling1D(2)(d5)
    d7 = Conv1D(64, 1, strides=1, activation='relu', padding='same')(d6)
    d8 = UpSampling1D(2)(d7)
    d9 = UpSampling1D(2)(d8)
    decoded = Conv1D(1, 1, strides=1, activation='sigmoid', padding='same')(d9)
    return Model(inputs=inputLayer, outputs=decoded)
