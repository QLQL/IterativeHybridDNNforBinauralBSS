# ITERATIVE DEEP NEURAL NETWORKS FOR SPEAKER-INDEPENDENT BINAURAL BLIND SPEECH SEPARATION
Qingju Liu, Yong Xu, Philip Coleman, Philip JB Jackson, Wenwu Wang 

[Paper](http://epubs.surrey.ac.uk/846225/) in ICASSP 2018 Calgary, Canada

An iterative deep neural network (DNN)-based binaural source separation scheme, for recovering two concurrent speech signals in a room environment. Besides the commonly-used spectral features, the DNN also takes non-linearly wrapped binaural spatial features as input, which are refined iteratively using parameters estimated from the DNN output via a feedback loop. Different DNN structures have been tested, including a classic multilayer perception regression architecture as well as a new hybrid network with both convolutional and densely-connected layers. 

******************************************************************************************
This code is implemented in Python, using [Keras](https://keras.io/) with [Theano](http://deeplearning.net/software/theano/) backend where "image_data_format": "channels_first" was set.


## Data preparation

To train the source separation models, you need to prepare the training data and two training parameters.

- Training data is organised in the folder /TrainDataDirectory that has three subfolders with different gender (male and female, M and F) combinations, i.e. /MM, /MF, /FF. In each subfolder, there are a large number of mat files (generated from Matlab), each containing the following training features:

`mixLogPower`------------257xFrame when FFT size is 512

The spectral features extracted from the binaural mixtures directly 

`mixAngle_L`-------------257xFrame

Optional, the angle information for the left-channel, only used to debug the code by applying inverse STFT to validate the results

`mixAngle_R`-------------257xFrame

Optional, the angle information for the right-channel, only used for debug

`s1LogPower`-------------257xFrame

The groundtruth spectral information for one target signal (speaker 1)

`s2LogPower`-------------257xFrame

The groundtruth spectral information for the second target signal (speaker 2)

- Normalisation parameters in a matfile generated using Matlab

`sig_mean`---------------257x1 

Mean of the mixture spectral features in the training data

`sig_std`----------------257x1 

std of the mixture spectral features in the training data

- IPD mean parameters

`IPD_mean`---------------257x5

The mean of frequency-dependent interaural phase difference (IPD) for one sound source from different direction, e.g. in our training scenario, we have considered the following 5 azimuths [-60,-30,0,30,60]


The feature extraction and parameter estimation process for each simulated binaural mixture is self-explanatory in the paper, which can be easily implemented in Matlab


## Train

Main code in BSSProcessFeature.py

## Test

Apply source separation after the DNN model is trained.

Example code in BSSseparationUnmatched.py


******************************************************************************************
# NOTE

This is a crude version of python implementation for the method presented in the paper, and the code can be optimised. For instance, the following Keras tricks can be exploited

+++++++++++++++

Instead of `train_on_batch` for each of generated batch of training data, we could directly use the `fit_generator`. In such condition, we need to have a proper data generator that **yield** data in a infinite loop rather than **return** only a batch of data.


+++++++++++++++

`callbacks` should be used for more flexible check up and convergence control, e.g. `CSVLogger`, `LearningRateScheduler`, `ModelCheckpoint` .etc

+++++++++++++++

Some very bad programming syntax spoiled by Matlab should be avoided. To slice a subset from a ndarray xyz of size 3x20x100, I often use xyz[0,:,:] instead of xyz[0], for instance.

