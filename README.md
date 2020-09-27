# Audio-Anomaly-Detection
Deep neural network based anomaly detection methods.
For Audio data, we can use this to do three things: 1. detect anomalies; 2. data quality measurement; 3. denoise.

### 1. Convolutional Autoencoder (CAE)
We can treat audio data in two ways: the original signal (1D) or the time-frequency domain signal (2D). If we convert audio signal into 2D spectrogram, it can be regarded as image to some extent. So why are the convolutional autoencoders suitable for image data? We see huge loss of information when slicing and stacking the data. Instead of stacking the data, the Convolution Autoencoders keep the spatial information of the input image data as they are, and extract information gently in what is called the Convolution layer. following figure demonstrates that a flat 2D image is extracted to a thick square (Conv1), then continues to become a long cubic (Conv2) and another longer cubic (Conv3). This process is designed to retain the spatial relationships in the data. This is the encoding process in an Autoencoder. In the middle, there is a fully connected autoencoder whose hidden layer is composed of only 10 neurons. After that comes with the decoding process that flattens the cubics, then to a 2D flat image. The encoder and the decoder are symmetric in following figure. Refer to: https://towardsdatascience.com/convolutional-autoencoders-for-image-noise-reduction-32fce9fc1763

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/ae_cnn.png)

Here are some details of our model design. Firstly, both encoder and decoder have 5 CNN layers and the number of filters is increasing and decreasing respectively, each cnn layer is followed by a BN layer and a MaxPool layer. The kernels are all square. Secondly, the loss function of CAE model we have 3 diferent choices: MSE, nRMSE and Structral Similarity. If you compile the model with different loss function, you will get different results. And because we are using autoencoder, we also need an index to represent the anomaly score (or to say resconstruction error), this index could be same with your loss. The expriment shows that using nRMSE as loss function and reconstruction error we can get the best result. Further more, on the basis of CAE model, we built Inception block which could be added to the original model.
In practice, if we are using audio data, the shape of the input data is determined by the parameters of time-frequency convertion (nfft and overlap in Stfts) and the duration of the audio itself. In our case, the time dimension is too large, so we applied PCA to reduce the time dimension to 256:

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/PCA.png)

If trained by AutoEncoder model, you have to make sure all of your training data (normal) have identical distribution. For normal sample and abnormal sample, the resconstruction performance is shown below (left: normal, right: abnormal. The reconstruction error in nRMSE metric is 0.20 and 0.69 respectively):

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/normal.png) ![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/abnormal.png)

#### Codes could be found in Conv_AE.py
