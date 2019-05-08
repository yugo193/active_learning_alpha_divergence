from model import build_model
import numpy as np

def train(alpha, input_shape, problem_type, model_name, epochs, batch_size, K_mc,x_train, y_train) :
  model = build_model(alpha, input_shape, problem_type, model_name, K_mc)
  y_train_dup = np.squeeze(np.concatenate(K_mc * [y_train[:, None]], axis=1))
  if(problem_type == "reg"):
    y_train_dup = np.reshape(y_train_dup, (y_train_dup.shape[0],y_train_dup.shape[1],1))
  model.fit(x_train, y_train_dup, epochs = epochs,batch_size=batch_size,verbose=2)
  return model
