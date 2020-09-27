#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Conv_AE.py
@Time    :   2020/09/27 14:48:14
@Author  :   Wang Minxuan 
@Version :   1.0.0
@Contact :   mx.wang@cyber-insight.com
@License :   (C)Copyright 2019-2020, CyberInsight
@Desc    :   None
'''

# here put the import lib
import os
import glob
import re
from tqdm import tqdm
import sys
import time
import numpy as np
import librosa
import librosa.core
import librosa.feature
from scipy import signal
import yaml
from keras.models import Model, Sequential
from keras.layers import *
from keras import regularizers
from keras.optimizers import adam
from keras import backend as K
import csv
import itertools
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from skimage.transform import resize
from keras.losses import binary_crossentropy
from keras import backend as K
import skimage.metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def loss_plot(loss, val_loss):
    ax = plt.figure(figsize=(30, 10)).add_subplot(1, 1, 1)
    ax.cla()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title("Model loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Train", "Validation"], loc="upper right")
def save_figure( name):
    plt.savefig(name)

def baseline_cae(inputDim_0, inputDim_1):
    #input
    inp = Input(shape=(inputDim_0, inputDim_1, 1))
    # encoder
    encoding_dim = (inputDim_0 // 32) * (inputDim_1 // 32)
    e = Conv2D(8, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(inp)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(16, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(32, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(64, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(128, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    l = Flatten()(e)
    l = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(l)
    l = LeakyReLU(alpha=0.1)(l)
    encoded = l
    #decoder
    d = Reshape(((inputDim_0 // 32), (inputDim_1 // 32), 1))(encoded)
    d = Conv2D(128, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(64, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(32, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(16, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(8, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(1, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(d)           
    d = BatchNormalization()(d)
    d = Activation('sigmoid')(d)
    decoded = d
    # model
    ae = Model(inp, decoded)
    return ae

def inception_layer(x, filters):
    # 1x1 convolution
    x0 = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x0 = BatchNormalization()(x0)
    x0 = LeakyReLU(alpha=0.1)(x0)
    # 3x3 convolution
    x1 = Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    # 5x5 convolution
    x2 = Conv2D(filters, (5,5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    # Max Pooling
    x3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    x3 = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    output = concatenate([x0, x1, x2, x3], axis = 3)
    return output

##### Inception-like Convolutional AutoEncoder #####

def inceptionCAE(img_dim, filters):
    # input
    input_img = Input(shape=img_dim) # adapt this if using `channels_first` image data format
    # encoder
    x = inception_layer(input_img, filters[0])
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = inception_layer(x, filters[1])
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = inception_layer(x, filters[2])
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = x
    #decoder
    x = inception_layer(x, filters[2])
    x = UpSampling2D((2, 2))(x)
    x = inception_layer(x, filters[1])
    x = UpSampling2D((2, 2))(x)
    x = inception_layer(x, filters[0])
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(img_dim[2], (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder

# Loss functtion
def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 2.0))

def nrmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) / K.sqrt(K.mean(K.square(y_true)))

def train(dataset_name='blade'):
    '''
	train the convolutional encoder-decoder anomaly detector, "blade" is a custom audio dataset, each sample is converted into
	image (log-energy spectrogram) based on STFTs, the time dimension is then reduced by PCA, resize technique is also employed 
	to get square input dimension.
	besides, the choices of loss function is listed, including mse (default), nrmse (best in our case) and stuctrual similarity (image)
    '''
    if os.path.exists('{}_feature.npy'.format(dataset_name)):
        train_data = np.load('{}_feature.npy'.format(dataset_name))
    else:
        if dataset_name == 'mnist':
            (X_train, y_train), (_, _) = mnist.load_data()
            # Make the data range between 0~1.
            X_train = X_train / 255
            specific_idx = np.where(y_train == self.attention_label)[0]
            _data = resize(X_train[specific_idx], (len(specific_idx), 256, 256))
            train_data = _data.reshape(-1, 256, 256, 1)
        elif dataset_name == 'blade':
            allFiles = glob.glob('./data/train' + '/*.wav')
            X_train = np.zeros((len(allFiles) * 256, 256))
            print(X_train.shape)
            for i, f in enumerate(allFiles):
                wav, sr = librosa.load(f, sr=None)
                wn = [2 * 1000.0 / sr, 0.99]
                b, a = signal.butter(8, wn, 'bandpass')
                wav = signal.filtfilt(b, a, wav)
                stft = np.abs(signal.stft(wav, fs=sr, window='hanning', nperseg=1024, noverlap=512)[2])
                pca_sk = PCA(n_components=512)
                stft = pca_sk.fit_transform(stft[:-1, :])
                db = librosa.amplitude_to_db(stft, ref=np.min, top_db=1000)
                db = np.flipud(resize(db, (256, 256)))
                normed_db = db / np.max(db)
                X_train[256 * i: 256 * (i + 1), :] = normed_db
            train_data = X_train.reshape(len(allFiles), 256, 256, 1)
        else:
            assert('Error in loading dataset')
        np.save('{}_feature.npy'.format(dataset_name), train_data)
    
    model_file_path = "{model}/model_{dataset_name}.hdf5".format(
            model='./model', dataset_name=dataset_name
            )
    history_img = "{model}/history_{dataset_name}.png".format(
        model='./model', dataset_name=dataset_name
        )
    print("============== MODEL TRAINING ==============")
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    model = baseline_cae(256, 256)
    model.summary()
    model.compile(optimizer='adam', loss=nrmse_loss) # adam, rmsprop; mean_squared_error
    history = model.fit(
        train_data, train_data, epochs=200, batch_size=8, shuffle=True, validation_split=0.1, verbose=1
        )
    loss_plot(history.history["loss"], history.history["val_loss"])
    save_figure(history_img)
    model.save(model_file_path)
    print("save_model -> {}".format(model_file_path))
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("============== END TRAINING ==============")
    print('start time: {}, end time: {}'.format(start_time, end_time))

def test(test=True, dataset_name='blade', label=2, data_index=10, _class='abnormal', metric='nrmse'):
    '''
	if using MNIST dataset, you can randomly set a label as "normal" class and others as "abnormal"
	a metric has to be determined to present the reconstruction loss, also known as "anomaly score"
    '''
    assert metric in ['binary_cross_entropy', 'structral_similarity', 'nrmse']
    
    model_file = "{model}/model_{dataset_name}.hdf5".format(
            model='./model', dataset_name=dataset_name
            )
    # load model file
    if not os.path.exists(model_file):
        raise Exception("{} model not found".format(machine_type))
    model = load_model(model_file)
    
    if dataset_name == 'mnist':
        assert label in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = X_train / 255
        specific_idx = np.where(y_train == label)[0]
        if data_index >= len(X_train):
            data_index = 0
        data = X_train[specific_idx].reshape(-1, 28, 28, 1)[data_index: data_index+1]
        test_data = resize(data, (1, 256, 256, 1))
    elif dataset_name == 'blade':
        if test is True:
            assert _class in ['normal', 'abnormal', 'validation', 'evaluation']
            allFiles = glob.glob('../../data/test/{}'.format(_class) + '/*.wav')
        else:
            allFiles = glob.glob('../../data/train' + '/*.wav')
        f = allFiles[data_index: data_index+1][0]
        wav, sr = librosa.load(f, sr=None)
        wn = [2 * 1000.0 / sr, 0.99]
        b, a = signal.butter(8, wn, 'bandpass')
        wav = signal.filtfilt(b, a, wav)
        stft = np.abs(signal.stft(wav, fs=sr, window='hanning', nperseg=512, noverlap=256)[2])
        pca_sk = PCA(n_components=256)
        stft = pca_sk.fit_transform(stft[:-1, :])
        db = librosa.amplitude_to_db(stft, ref=np.min)
        normed_db = db / np.max(db)
        test_data = normed_db.reshape(1, 256, 256, 1)

    model_predicts = model.predict(test_data)
#     print(model_predicts.shape)
    
#     fig = plt.figure(figsize=(8, 8))
#     columns = 1
#     rows = 2
#     fig.add_subplot(rows, columns, 1)
    input_image = test_data.reshape((256, 256))
    reconstructed_image = model_predicts.reshape((256, 256))
#     plt.title('Input')
#     plt.imshow(input_image, label='Input')
#     fig.add_subplot(rows, columns, 2)
#     plt.title('Reconstruction')
#     plt.imshow(reconstructed_image, label='Reconstructed')
#     plt.show()
    # Compute the mean binary_crossentropy loss of reconstructed image.
    y_true = K.variable(input_image)
    y_pred = K.variable(reconstructed_image)
    if metric == 'binary_cross_entropy':
        error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
    elif metric == 'structral_similarity':
        error = 1 - skimage.metrics.structural_similarity(input_image, reconstructed_image)
    elif metric == 'nrmse':
        error = np.sqrt(mean_squared_error(input_image, reconstructed_image)) / np.sqrt(np.mean(input_image**2))
    print('Reconstruction loss:', error)
    return error