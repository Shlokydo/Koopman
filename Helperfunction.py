import numpy as np
from scipy import integrate
import tensorflow as tf
import pandas as pd
import pickle

import os
import time
import datetime
from pathlib import Path

#Code for importing the Dataset from the HDF5 file
def import_datset(dataset, key):
    dataframe = pd.read_hdf(dataset, key = key)
    return dataframe

#Code for converting the dataframe into numpy arrays
def dataframe_to_nparray(dataframe):
    return dataframe.values

#Code for resizing the 2D numpy array to 3D array, where each row corresponds to a set of bifurication parameters
def nparray_resize_2Dto3D(nparray):    
    rows = len(np.unique(nparray[:,1]))
    columns = int(nparray.shape[0]/rows)
    height = int(nparray.shape[1])
    
    nparray.resize(rows, columns, height)
    
    #Deleting the height vector containg data related Iteration number
    nparray = np.delete(nparray, 0, axis= 2)
    
    return nparray

def nparray_del_vec_along_axis(nparray, vec, ax):
    return np.delete(nparray, vec, axis = ax)

def change_nparray_datatype(nparray, datatype):
    return nparray.astype(datatype)

def dataset_scaling(dataset, scaler):
    return dataset * float(scaler)

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i*n_steps + n_steps - 1
        if (end_ix > len(sequences) - 1):
            break
        seq_x, seq_y = sequences[i*n_steps:end_ix, :], sequences[i*n_steps + 1:end_ix + 1, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def sequences_test(sequences, n_steps):
    X = list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i*n_steps + n_steps
        if (end_ix > len(sequences) - 1):
            break
        seq_x = sequences[i*n_steps:end_ix, :]
        X.append(seq_x)
    return np.array(X)

def np_array_shuffle(initial_dataset_x, initial_dataset_y):
    initial_dataset_concat = np.concatenate((initial_dataset_x, initial_dataset_y), axis = 2)
    np.random.shuffle(initial_dataset_concat)
    initial_dataset_x, initial_dataset_y = np.split(initial_dataset_concat, 2, axis= 2)
    return initial_dataset_x, initial_dataset_y

#Code for creating Tensorflow Dataset:
def create_tfdataset(initial_dataset):
    tf_dataset = tf.data.Dataset.from_tensor_slices(initial_dataset)
    return tf_dataset

def write_dataframe(dataframe, filename):   
    dataframe.to_csv(filename)

def read_dataframe(filename):
    return pd.read_csv(filename).to_dict(orient= 'records')[0]

def write_pickle(dicty, filename):
    pickle_out = open(filename, "wb")
    pickle.dump(dicty, pickle_out)
    pickle_out.close()

def read_pickle(filename):
    pickle_in = open(filename, "rb")
    pl = pickle.load(pickle_in)
    pickle_in.close()
    return pl

def input_generator(input_value):
    input_value = np.asarray(input_value, dtype= np.float32)
    input_value = np.expand_dims(input_value, axis=0)
    input_value = np.expand_dims(input_value, axis=0)
    return input_value

def nl_pendulum(extension_list, time):

    def nl_pendulum_deriv(x_y, t0):
        """Compute the time-derivative."""
        x, y = x_y
        return [y, -np.sin(x)]

    x = extension_list
    # Solve for the trajectories
    
    x_t_new = []
    for x0i in x:
        x_t_new.append(integrate.odeint(nl_pendulum_deriv, x0i, time))
            
    x_t_new = np.asarray(x_t_new)

    return x_t_new

def discrete_solve(N = 10, mu = 5, _lambda = 2, max_time = 30, delta_t = 0.02, x0 = [-0.5, 0.5], x1 = [-0.5 , 0.5]): 
    
    x0 = np.resize(np.random.uniform(x0[0], x0[1], N),(N,1))
    x0 = np.insert(x0, 1, np.resize(np.random.uniform(x1[1], x1[0], N),(N,1)).T, axis=1)

    def discrete_deriv(x_y, t0, mu = mu, _lambda = _lambda):
        """Compute the time-derivative of a Lorenz system."""
        x, y = x_y
        return [mu * x, _lambda * (y - (x*x))]

    time = np.arange(0, max_time+delta_t, delta_t)
    x_t = np.asarray([integrate.odeint(discrete_deriv, x0i, time) for x0i in x0])

    return time, x_t

def difference(x_true, x_pred):
    diff = x_true - x_pred
    per = np.abs((diff / x_true) * 100)
    #print(per)
    per = np.log10(per)
    #print(per)
    per[per == -np.inf] = 0
    return per

def write_to_json(loc, model):
    with open(loc, 'w') as json_file:
        json_file.write(model)

def read_json(loc):
    json_file = open(loc, 'r')
    content = json_file.read()
    json_file.close()
    return content
