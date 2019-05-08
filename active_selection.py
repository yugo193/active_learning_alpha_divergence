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


def active_selection(model, unlabelled_data, nb_query, mean_X_train, std_X_train,problem_type, K_mc, batch_size):
  if(problem_type == "reg"):
    n = min(nb_query*20, len(unlabelled_data[0]))
  else:
    n = 2000
  subset_index = np.random.permutation(len(unlabelled_data[0]))
  subset = unlabelled_data[0][subset_index[:n]]
  subset = (subset - mean_X_train)/std_X_train
  dropout_iterations = 50
  if(problem_type == "reg"):
    all_score = []
    for i in range(dropout_iterations//K_mc):
      dropout_score = model.predict(subset,batch_size=batch_size, verbose=0)
      dropout_score = np.reshape(dropout_score, dropout_score.shape[:-1])
      all_score.append(dropout_score)
    dropout_all = np.concatenate(all_score,axis=1)
    dropout_mean = np.mean(dropout_all,1,keepdims=True)
    dropout_variance = np.mean((dropout_all-dropout_mean)**2,1)
    index = np.argsort(dropout_variance)[::-1]
  else:
    for d in range(dropout_iterations//K_mc):
      dropout_score = model.predict(X_Pool_Dropout,batch_size=batch_size, verbose=0)
      score_All = score_All + np.sum(dropout_score,axis=1)
      dropout_score_log = np.log2(dropout_score)
      Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
      Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=(1,2))
      All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout
    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)
    dropout_bald = Entropy_Average_Pi - Average_Entropy
    index = np.argsort(dropout_bald.flatten())[::-1]
  index_query = subset_index[index[:nb_query]]
  index_unlabelled = subset_index[index[nb_query:]]

  new_data = unlabelled_data[0][index_query,:]
  new_labels = unlabelled_data[1][index_query]
  return (new_data, new_labels), \
    (np.concatenate([unlabelled_data[0][index_unlabelled], unlabelled_data[0][subset_index[n:]]], axis=0), np.concatenate([unlabelled_data[1][index_unlabelled], unlabelled_data[1][subset_index[n:]]], axis=0))
