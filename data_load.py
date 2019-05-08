from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras import losses
from keras.utils import np_utils, generic_utils
import numpy as np
import scipy as sp
import random
import scipy.io
from scipy.stats import mode

img_rows, img_cols = 28, 28

def build_data(data_name, model_name):
  data_x = []
  data_y = []
  dataset = ["MNIST","White","CT","Red", "CASP", "Concrete", "News", "KEGG"]
  if(data_name not in dataset):
    raise ValueError("Not found")
  if(data_name == "MNIST"):
    nb_classes = 10
    nb_query = 10
    (X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()
    if(model_name == "fn"):
      X_train_All = X_train_All.reshape(X_train_All.shape[0],img_rows*img_cols)
      X_test = X_test.reshape(X_test.shape[0],img_rows*img_cols)
      input_shape = (img_rows*img_cols,)
    else:
      if K.image_data_format() == 'channels_first':
        X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
      else:
        X_train_All = X_train_All.reshape(X_train_All.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

    if(model_name=="fn"):

      X_train_All = X_train_All[random_split, :]
      y_train_All = y_train_All[random_split]


      X_valid = X_train_All[10000:15000, :]
      y_valid = y_train_All[10000:15000]

      X_Pool = X_train_All[20000:60000, :]
      y_Pool = y_train_All[20000:60000]

      X_train_All = X_train_All[0:10000, :]
      y_train_All = y_train_All[0:10000]


      #training data to have equal distribution of classes
      idx_0 = np.array( np.where(y_train_All==0)  ).T
      idx_0 = idx_0[0:5,0]
      X_0 = X_train_All[idx_0, :]
      y_0 = y_train_All[idx_0]

      idx_1 = np.array( np.where(y_train_All==1)  ).T
      idx_1 = idx_1[0:5,0]
      X_1 = X_train_All[idx_1, :]
      y_1 = y_train_All[idx_1]

      idx_2 = np.array( np.where(y_train_All==2)  ).T
      idx_2 = idx_2[0:5,0]
      X_2 = X_train_All[idx_2, :]
      y_2 = y_train_All[idx_2]

      idx_3 = np.array( np.where(y_train_All==3)  ).T
      idx_3 = idx_3[0:5,0]
      X_3 = X_train_All[idx_3, :]
      y_3 = y_train_All[idx_3]

      idx_4 = np.array( np.where(y_train_All==4)  ).T
      idx_4 = idx_4[0:5,0]
      X_4 = X_train_All[idx_4, :]
      y_4 = y_train_All[idx_4]

      idx_5 = np.array( np.where(y_train_All==5)  ).T
      idx_5 = idx_5[0:5,0]
      X_5 = X_train_All[idx_5, :]
      y_5 = y_train_All[idx_5]

      idx_6 = np.array( np.where(y_train_All==6)  ).T
      idx_6 = idx_6[0:5,0]
      X_6 = X_train_All[idx_6, :]
      y_6 = y_train_All[idx_6]

      idx_7 = np.array( np.where(y_train_All==7)  ).T
      idx_7 = idx_7[0:5,0]
      X_7 = X_train_All[idx_7, :]
      y_7 = y_train_All[idx_7]

      idx_8 = np.array( np.where(y_train_All==8)  ).T
      idx_8 = idx_8[0:5,0]
      X_8 = X_train_All[idx_8, :]
      y_8 = y_train_All[idx_8]

      idx_9 = np.array( np.where(y_train_All==9)  ).T
      idx_9 = idx_9[0:5,0]
      X_9 = X_train_All[idx_9, :]
      y_9 = y_train_All[idx_9]

    else:
      X_train_All = X_train_All[random_split, :, :, :]
      y_train_All = y_train_All[random_split]


      X_valid = X_train_All[10000:15000, :, :, :]
      y_valid = y_train_All[10000:15000]

      X_Pool = X_train_All[20000:60000, :, :, :]
      y_Pool = y_train_All[20000:60000]

      X_train_All = X_train_All[0:10000, :, :, :]
      y_train_All = y_train_All[0:10000]


      #training data to have equal distribution of classes
      idx_0 = np.array( np.where(y_train_All==0)  ).T
      idx_0 = idx_0[0:5,0]
      X_0 = X_train_All[idx_0, :, :, :]
      y_0 = y_train_All[idx_0]

      idx_1 = np.array( np.where(y_train_All==1)  ).T
      idx_1 = idx_1[0:5,0]
      X_1 = X_train_All[idx_1, :, :, :]
      y_1 = y_train_All[idx_1]

      idx_2 = np.array( np.where(y_train_All==2)  ).T
      idx_2 = idx_2[0:5,0]
      X_2 = X_train_All[idx_2, :, :, :]
      y_2 = y_train_All[idx_2]

      idx_3 = np.array( np.where(y_train_All==3)  ).T
      idx_3 = idx_3[0:5,0]
      X_3 = X_train_All[idx_3, :, :, :]
      y_3 = y_train_All[idx_3]

      idx_4 = np.array( np.where(y_train_All==4)  ).T
      idx_4 = idx_4[0:5,0]
      X_4 = X_train_All[idx_4, :, :, :]
      y_4 = y_train_All[idx_4]

      idx_5 = np.array( np.where(y_train_All==5)  ).T
      idx_5 = idx_5[0:5,0]
      X_5 = X_train_All[idx_5, :, :, :]
      y_5 = y_train_All[idx_5]

      idx_6 = np.array( np.where(y_train_All==6)  ).T
      idx_6 = idx_6[0:5,0]
      X_6 = X_train_All[idx_6, :, :, :]
      y_6 = y_train_All[idx_6]

      idx_7 = np.array( np.where(y_train_All==7)  ).T
      idx_7 = idx_7[0:5,0]
      X_7 = X_train_All[idx_7, :, :, :]
      y_7 = y_train_All[idx_7]

      idx_8 = np.array( np.where(y_train_All==8)  ).T
      idx_8 = idx_8[0:5,0]
      X_8 = X_train_All[idx_8, :, :, :]
      y_8 = y_train_All[idx_8]

      idx_9 = np.array( np.where(y_train_All==9)  ).T
      idx_9 = idx_9[0:5,0]
      X_9 = X_train_All[idx_9, :, :, :]
      y_9 = y_train_All[idx_9]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_Pool = X_Pool.astype('float32')
    X_train /= 255
    X_valid /= 255
    X_Pool /= 255
    X_test /= 255
    input_shape = X_train.shape[1:]

    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    labelled_data = (X_train, Y_train)
    unlabelled_data = (X_Pool, Y_Pool)
    test_data = (X_test, Y_test)
  else:

    if(data_name == "White"):
      input_dim=11
      f = open('data/winequality-white.csv')
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+1):
          break
        data_x.append(data_list[:-1])
        data_y.append(data_list[-1])
        input_shape=(input_dim,)
    if(data_name == "CT"):
      input_dim=384
      f = open('data/slice_localization_data.csv')
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+1):
          break
        data_x.append(data_list[:-1])
        data_y.append(data_list[-1])
      input_shape=(input_dim,)
    if(data_name == "Red"):
      input_dim=11
      f = open('data/winequality-red.csv')
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+1):
          break
        data_x.append(data_list[:-1])
        data_y.append(data_list[-1])
      input_shape=(input_dim,)
    if(data_name == "CASP"):
      input_dim=9
      f = open('data/CASP.csv')
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+1):
          break
        data_x.append(data_list[1:])
        data_y.append(data_list[0])
      input_shape=(input_dim,)
    if(data_name == "Concrete"):
      input_dim=8
      f = open('data/Concrete_Data.csv',encoding="utf-8_sig")
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+1):
          break
        data_x.append(data_list[:-1])
        data_y.append(data_list[-1])
      input_shape=(input_dim,)
    if(data_name == "News"):
      input_dim=59
      f = open('data/OnlineNewsPopularity.csv')
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+1):
          break
        data_x.append(data_list[:-1])
        data_y.append(data_list[-1])
      input_shape=(input_dim,)
    if(data_name == "KEGG"):
      input_dim=22
      f = open('data/Relation Network (Directed).data')
      data1 = f.read()
      f.close()
      lines1 = data1.split('\n')
      data_x = []
      data_y = []
      for line in lines1:
        data_list = line.split(',')
        if(len(data_list) != input_dim+2):
          break
        data_x.append(data_list[1:-1])
        data_y.append(data_list[-1])
      input_shape=(input_dim,)
    data_x = np.array(data_x,dtype="float32")
    data_y = np.array(data_y, dtype="float32")
    random_pool_split = np.asarray(random.sample(range(0,data_x.shape[0]),data_x.shape[0]))
    data_x = data_x[random_pool_split,:]
    data_y = data_y[random_pool_split]
    test_data_x = data_x[:data_x.shape[0]//5,:]
    test_data_y = data_y[:data_x.shape[0]//5]
    unlabelled_data_x = data_x[data_x.shape[0]//5:-data_x.shape[0]//10,:]
    unlabelled_data_y = data_y[data_x.shape[0]//5:-data_x.shape[0]//10]
    labelled_data_x = data_x[-data_x.shape[0]//10:,:]
    labelled_data_y = data_y[-data_x.shape[0]//10:]
    nb_query = data_x.shape[0]*3//100
    test_data = (test_data_x, test_data_y)
    unlabelled_data = (unlabelled_data_x, unlabelled_data_y)
    labelled_data = (labelled_data_x, labelled_data_y)
  return (labelled_data, unlabelled_data, test_data, nb_query,input_shape)
