# this is one RNN cell i.e for for one time stamp.

import numpy as np
from rnn_utils import *


def rnn_cell_forward(xt, a_prev, parameters):
    """
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    n_x  denote the number of units in a single timestep of a single training example i.e dictionary size
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"] # numpy array of shape (n_a, n_x)
    Waa = parameters["Waa"] # numpy array of shape (n_a, n_a)
    Wya = parameters["Wya"] # numpy array of shape (n_y, n_a)
    ba = parameters["ba"] # numpy array of shape (n_a, 1)
    by = parameters["by"] # numpy array of shape (n_y, 1)
    
    ### START CODE HERE ### (≈2 lines)
    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya, a_next) + by)   
    ### END CODE HERE ###
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache

np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
print("a_next_tmp[4] = ", a_next_tmp[4])
print("a_next_tmp.shape = ", a_next_tmp.shape)
print("yt_pred_tmp[1] =", yt_pred_tmp[1])
print("yt_pred_tmp.shape = ", yt_pred_tmp.shape)