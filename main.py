from data_load import build_data
from train import train
from active_selection import active_selection
import numpy as np
from test import test_reg, test_class

#import test
#import model

data_name = "Concrete"
centering = True
problem_type = "reg"
model_name = "shallow"
epochs = 10
batch_size = 10
alpha = 1.0
K_mc = 10


def main():
  labelled_data, unlabelled_data, test_data,nb_query, input_shape = build_data(data_name, model_name)
  percentage_data = len(labelled_data[0])
  N_pool = len(labelled_data[0]) + len(unlabelled_data[0])
  print('START')
  i=0
  while( percentage_data<=N_pool):
    X_train, Y_train = labelled_data
    if(centering):
      std_X_train = np.std(X_train, 0)
      std_X_train[ std_X_train == 0 ] = 1
      mean_X_train = np.mean(X_train, 0)
      centerized_X_train = (X_train - mean_X_train) / std_X_train
    else:
      std_X_train = 1.
      mean_X_train = 0.
      centerized_X_train = X_train
    model = train(alpha, input_shape, problem_type, model_name, epochs, batch_size, K_mc, centerized_X_train, Y_train)
    query, unlabelled_data = active_selection(model, unlabelled_data, nb_query, mean_X_train, std_X_train, problem_type, K_mc, batch_size)
    if(problem_type == "class") :
      acc, ll = test_class(test_model, X_test, Y_test, mean_X_train, std_X_train)
    if(problem_type == "reg"):
      rmse, mae, maxae = test_reg(model,test_data[0],test_data[1],mean_X_train,std_X_train,K_mc)
    labelled_data_0 = np.concatenate((labelled_data[0], query[0]), axis=0)
    labelled_data_1 = np.concatenate((labelled_data[1], query[1]), axis=0)
    labelled_data = (labelled_data_0, labelled_data_1)
    percentage_data +=nb_query

main()
