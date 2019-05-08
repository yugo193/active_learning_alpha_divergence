from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras import losses
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy as sp
import random
import scipy.io
from scipy.stats import mode

def test_reg(model, X, Y, mean_X_train, std_X_train, K_mc):
    X = (X - mean_X_train)/std_X_train
    pred = model.predict(X)
    if(K_mc != 1):
        pred = np.reshape(pred, pred.shape[:-1])
    pred = np.mean(pred, 1)
    rmse = np.sqrt(np.mean((Y-pred)**2))
    mae = np.mean(np.abs(Y-pred))
    maxae = np.max(np.abs(Y-pred))
    return rmse, mae, maxae

def test_class(model, X, Y):
    pred = model.predict(X)  # N x K x D
    pred = np.mean(pred, 1)
    acc = np.mean(np.argmax(pred, axis=-1) == np.argmax(Y, axis=-1))
    ll = np.sum(np.log(np.sum(pred * Y, -1)))
    return acc, ll
