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

def apply_layers(inp, layers):
  output = inp
  for layer in layers:
    output = layer(output)
  return output


def GenerateMCSamples(inp, layers, K_mc):
  if K_mc == 1:
    return apply_layers(inp, layers)
  output_list = []
  for _ in range(K_mc):
    output_list += [apply_layers(inp, layers)]
  def pack_out(output_list):
    output = K.stack(output_list)
    return K.permute_dimensions(output, (1, 0, 2))
  def pack_shape(s):
    s = s[0]
    return (s[0], K_mc, s[1])
  out = Lambda(pack_out, output_shape=pack_shape)(output_list)
  return out

def logsumexp(x, axis=None):
  x_max = K.max(x, axis=axis, keepdims=True)
  return K.log(K.sum(K.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def bbalpha_softmax_cross_entropy_with_mc_logits(alpha):
  alpha = K.cast_to_floatx(alpha)
  def loss(y_true, mc_logits):
    mc_log_softmax = mc_logits - K.max(mc_logits, axis=2, keepdims=True)
    mc_log_softmax = mc_log_softmax - K.log(K.sum(K.exp(mc_log_softmax), axis=2, keepdims=True))
    mc_ll = K.sum(y_true * mc_log_softmax, -1)
    K_mc = mc_ll.get_shape().as_list()[1]
    return - 1. / alpha * (logsumexp(alpha * mc_ll, 1) + K.log(1.0 / K_mc))
  return loss

def bbalpha_l2_loss(alpha):
  alpha = K.cast_to_floatx(alpha)
  tau = 1
  def loss(y_true,y_predict):
    mc_ll = tau/2*(y_true - y_predict)**2
    K_mc = mc_ll.get_shape().as_list()[1]  # only for tensorflow
    return - 1. / alpha * (logsumexp(-alpha * mc_ll, 1) + K.log(1.0 / K_mc))
  return loss


def bbalpha_softmax_cross_entropy_with_mc_logits(alpha):
  alpha = K.cast_to_floatx(alpha)
  def loss(y_true, mc_logits):
    # log(p_ij), p_ij = softmax(logit_ij)
    #assert mc_logits.ndim == 3
    mc_log_softmax = mc_logits - K.max(mc_logits, axis=2, keepdims=True)
    mc_log_softmax = mc_log_softmax - K.log(K.sum(K.exp(mc_log_softmax), axis=2, keepdims=True))
    mc_ll = K.sum(y_true * mc_log_softmax, -1)  # N x K
    K_mc = mc_ll.get_shape().as_list()[1]  # only for tensorflow
    return - 1. / alpha * (logsumexp(alpha * mc_ll, 1) + K.log(1.0 / K_mc))
  return loss


def get_deep_layers(input_dim):
  Weight_Decay=1e-5

  layers = []
  layers.append(Dense(256,kernel_regularizer=regularizers.l2(Weight_Decay)))
  layers.append(LeakyReLU(0.01))
  layers.append(Dropout(0.5))
  layers.append(Dense(128,kernel_regularizer=regularizers.l2(Weight_Decay)))
  layers.append(LeakyReLU(0.01))
  layers.append(Dropout(0.5))
  layers.append(Dense(64,kernel_regularizer=regularizers.l2(Weight_Decay)))
  layers.append(LeakyReLU(0.01))
  layers.append(Dropout(0.5))
  layers.append(Dense(1,kernel_regularizer=regularizers.l2(Weight_Decay)))
  return layers

def get_shallow_layers(input_dim):
  Weight_Decay=1e-5
  layers = []
  layers.append(Dense(50,kernel_regularizer=regularizers.l2(Weight_Decay),activation="relu"))
  layers.append(Dropout(0.5))
  layers.append(Dense(1,kernel_regularizer=regularizers.l2(Weight_Decay)))
  return layers


def get_cnn_layers():
  c = 3.5
  Weight_Decay = c / float(X_train.shape[0])

  layers = []
  layers.append(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
  layers.append(Conv2D(64, (3, 3), activation='relu'))
  layers.append(MaxPooling2D(pool_size=(2, 2)))
  layers.append(Dropout(0.25))
  layers.append(Flatten())
  layers.append(Dense(128, kernel_regularizer = regularizers.l2(Weight_Decay),activation='relu'))
  layers.append(Dropout(0.5))
  layers.append(Dense(nb_classes,kernel_regularizer = regularizers.l2(Weight_Decay)))
  return layers

def get_fn_layers():
  c = 3.5
  wd = c / float(X_train.shape[0])
  layers = []
  layers.append(Dropout(0.5))
  layers.append(Dense(100,input_dim=img_cols*img_rows,activation='relu', kernel_regularizer=regularizers.l2(wd)))
  layers.append(Dropout(0.5))
  layers.append(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(wd)))
  layers.append(Dropout(0.5))
  layers.append(Dense(nb_classes, kernel_regularizer=regularizers.l2(wd)))
  return layers

def build_model(alpha, input_shape, problem_type, model_name, K_mc):
  if(problem_type not in ["class", "reg"]):
    raise ValueError("Not found")
  if(problem_type == "class"):
    nb_test_mc = 100
    inp = Input(shape=input_shape)
    if(model_name not in ["cnn", "fn"]):
      raise ValueError("Not found")
    if(model_name == "fn"):
      layers = get_fn_layers()
    if(model_name == "cnn"):
      layers = get_cnn_layers()
    mc_logits = GenerateMCSamples(inp, layers, K_mc)
    if alpha != 0:
      model = Model(input=inp, output=mc_logits)
      model.compile(optimizer='sgd', loss=bbalpha_softmax_cross_entropy_with_mc_logits(alpha),metrics=['accuracy'])
    else:
      mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
      model = Model(input=inp, output=mc_softmax)
      model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    mc_logits = GenerateMCSamples(inp, layers, nb_test_mc)
    mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
    test_model = Model(input=inp, output=mc_softmax)

  if(problem_type == "reg"):
    inp = Input(shape=input_shape)
    if(model_name not in ["shallow", "deep"]):
      raise ValueError("Not found")
    if(model_name == "shallow"):
      layers = get_shallow_layers(input_shape)
    if(model_name == "deep"):
      layers = get_deep_layers(input_shape)
    mc_logits = GenerateMCSamples(inp, layers, K_mc)
    if alpha != 0:
      model = Model(input=inp, output=mc_logits)
      model.compile(optimizer='adam', loss=bbalpha_l2_loss(alpha),metrics=['mse'])
    else:
      model = Model(input=inp, output=mc_logits)
      model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
  return model
