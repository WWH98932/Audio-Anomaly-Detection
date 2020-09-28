# Audio-Anomaly-Detection
Deep neural network based anomaly detection methods.
For Audio data, we can use this to do three things: 1. detect anomalies; 2. measure data quality; 3. denoise.
### 1. 1D Dense/Convolutional Autoencoder
An autoencoder is a special type of neural network that copies the input values to the output values. If the number of neurons in the hidden layers is less than that of the input layers, the hidden layers will extract the essential information of the input values. This condition forces the hidden layers to learn the most patterns of the data and ignore the “noises”. That is the reason why autoencoders could be used to detect anomalies by just comparing the difference between the output and the input. The encoding process compresses the input values to get to the core layer. The decoding process reconstructs the information to produce the outcome. The decoding process mirrors the encoding process in the number of hidden layers and neurons. Most practitioners just adopt this symmetry.

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/ae_model_2.png)

In this part, we use 1D array (the original audio sequence) as input, a simple dense ae (like MLP) and a 1d convolution ae is implemented.
#### Codes could be found in 1d_AE.py

### 2. 2D Convolutional Autoencoder (CAE)
We can treat audio data in two ways: the original signal (1D) or the time-frequency domain signal (2D). If we convert audio signal into 2D spectrogram, it can be regarded as image to some extent. So why are the convolutional autoencoders suitable for image data? We see huge loss of information when slicing and stacking the data. Instead of stacking the data, the Convolution Autoencoders keep the spatial information of the input image data as they are, and extract information gently in what is called the Convolution layer. following figure demonstrates that a flat 2D image is extracted to a thick square (Conv1), then continues to become a long cubic (Conv2) and another longer cubic (Conv3). This process is designed to retain the spatial relationships in the data. This is the encoding process in an Autoencoder. In the middle, there is a fully connected autoencoder whose hidden layer is composed of only 10 neurons. After that comes with the decoding process that flattens the cubics, then to a 2D flat image. The encoder and the decoder are symmetric in following figure. Refer to: https://towardsdatascience.com/convolutional-autoencoders-for-image-noise-reduction-32fce9fc1763

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/ae_cnn.png)

Here are some details of our model design. Firstly, both encoder and decoder have 5 CNN layers and the number of filters is increasing and decreasing respectively, each cnn layer is followed by a BN layer and a MaxPool layer. The kernels are all square. Secondly, the loss function of CAE model we have 3 diferent choices: MSE, nRMSE and Structral Similarity. If you compile the model with different loss function, you will get different results. And because we are using autoencoder, we also need an index to represent the anomaly score (or to say resconstruction error), this index could be same with your loss. The expriment shows that using nRMSE as loss function and reconstruction error we can get the best result. Further more, on the basis of CAE model, we built Inception block which could be added to the original model.
In practice, if we are using audio data, the shape of the input data is determined by the parameters of time-frequency convertion (nfft and overlap in Stfts) and the duration of the audio itself. In our case, the time dimension is too large, so we applied PCA to reduce the time dimension to 256:

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/PCA.png)

If trained by AutoEncoder model, you have to make sure all of your training data (normal) have identical distribution. For normal sample and abnormal sample, the resconstruction performance is shown below (left: normal, right: abnormal. The reconstruction error in nRMSE metric is 0.20 and 0.69 respectively):

![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/normal.png) ![Image_text](https://github.com/WWH98932/Audio-Anomaly-Detection/blob/master/image/abnormal.png)

#### Codes could be found in Conv_AE.py
### 3. Anomaly Detection in GAN
As another kind of generative model (besides AE), GAN is also used in anomaly detection. Here is how it works:
Trainging stage: The network only learns the distribution of normal data, the model G can only reproduce/reconstruct normal data;
Testing stage: Pass the data into the model G, if the output is similar/identical with the input after reproduce/reconstruction, the input data is classificied as normal class.
The model G can be either GAN or AutoEncoder.
#### anoGAN - a DCGAN based anomaly detector
During training stage, a generator G is trained adversarially to reproduce the normal class from a random noise Z in latent space. In testing stage, randomly select a noise vector z from latent space, we can get the normal-like representation G(z) by feeding it into generator G, G is not trainable in this stage, all the parameters are fixed. But now the model still needs to be trained, we update the noise z iteratively, by comparing the output G(z) and testing data x to get a normal representation which is as similar as possible with x in ideal condition. So if the testing data x belongs to normal class, the output we get after iteration will be identical with x.

