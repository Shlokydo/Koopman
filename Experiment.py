import copy
import random
import numpy as np
import pandas as pd
import random as r
import os
import sys
import pickle
import shutil
import argparse
import time

import multigpu_training_test as multi_train
import Helperfunction as helpfunc
import tensorflow as tf
import valtest as testing

parser = argparse.ArgumentParser(description='Experiment controller')
parser.add_argument("--t", default='train', type=str, help="Training or testing flag setter", choices=["train", "test"])
parser.add_argument("--gpu", default=1, type=int, help="To enable/disable multi-gpu training. Default, true.", choices=[0, 1])
parser.add_argument("--epoch", default=200, type=int, help="Set the number of epochs.")
parser.add_argument("--experiment", "--exp", default="d", help="Name of the experiment(s)", nargs='*')
parser.add_argument("--key", default="nl_pendulum", help="Key for the dataset to be used from the HDF5 file")
parser.add_argument("--dataset", default="Dataset", help="Name of the .h5 dataset to be used for training")
parser.add_argument("--delta_t", default=0.2, type=float, help="Time stepping")
parser.add_argument("--num_ts", default=51, type=float, help="num of time steps")

args = parser.parse_args()

parameter_list = {}

#Basic setting for all the experiments
parameter_list['key'] = args.key
parameter_list['num_timesteps'] = 51   #Next version, need to read it from the dataset
parameter_list['num_validation_points'] = 1024
parameter_list['input_scaling'] = 1
parameter_list['inputs'] = 2
parameter_list['dataset'] = args.dataset + '.h5'

parameter_list['num_real'] = 0                          #Number of real Koopman eigenvalues
parameter_list['num_complex_pairs'] = 1                 #Number of complex conjugate eigenvalues
parameter_list['num_evals'] = parameter_list['num_real'] + 2 * parameter_list['num_complex_pairs']

parameter_list['experiments'] = args.experiment
parameter_list['checkpoint_expdir'] = './{}'.format(args.key)
parameter_list['checkpoint_dir'] = parameter_list['checkpoint_expdir']

#Getting the default parameter_list
#Settings related to dataset creation
parameter_list['Batch_size'] = 1024*4                      #Batch size
parameter_list['Batch_size_val'] = 1024*2                      #Batch size
parameter_list['Buffer_size'] = 50000                    #Buffer size for shuffle

#Encoder layer
parameter_list['en_units'] = 25                         #Number of neurons in the encoder lstm layer
parameter_list['en_width'] = 2                          #Number of lstm layers
parameter_list['en_activation'] = 'relu'                #All same as encoder till here
parameter_list['en_initializer'] = 'glorot_uniform'

#Koopman auxilary network
parameter_list['kaux_units'] = 5                                                     #Number of neurons in dense layers
parameter_list['kaux_width'] = 2                                                      #Number of dense layers
parameter_list['kaux_output_units_real'] = parameter_list['num_real']                 #Number of real outputs
parameter_list['kaux_output_units_complex'] = parameter_list['num_complex_pairs'] * 2 #Number of complex outputs
parameter_list['kp_initializer'] = 'glorot_uniform'     #Initializer of layers
parameter_list['kp_activation'] = 'tanh'                #Activation of layers
parameter_list['stateful'] = False

#Decoder layer
parameter_list['de_units'] = 25
parameter_list['de_width']  = 2
parameter_list['de_initializer'] = 'glorot_uniform'
parameter_list['de_activation'] = 'relu'                #All same as encoder till here
parameter_list['de_output_units'] = 2                   #Number of final output units

#Settings related to trainig
parameter_list['learning_rate'] = 0.001                 #Initial learning rate
parameter_list['lr_decay_rate'] = 0.96
parameter_list['learning_rate'] = parameter_list['learning_rate'] * parameter_list['Batch_size'] / 256.0
parameter_list['lr_decay_steps'] = 500000                 #No of steps for learning rate decay scheduling
parameter_list['dropout'] = 0.0                         #Dropout for the layers
parameter_list['early_stop_patience'] =21900               #Patience in num of epochs for early stopping
parameter_list['mth_step'] = 40                         #mth step for which prediction needs to be made
parameter_list['mth_cal_patience'] = 10                  #number of epochs after which mth loss is calculated
parameter_list['mth_no_cal_epochs'] = 50                #Number of epochs for which mth loss is not calculated
parameter_list['reconst_hp'] = 0.001
parameter_list['global_epoch'] = 0
parameter_list['val_min'] = 100

#Settings related to saving checkpointing and logging
parameter_list['max_checkpoint_keep'] = 4               #Max number of checkpoints to keep
parameter_list['num_epochs_checkpoint'] = 2            #Num of epochs after which to create a checkpoint
parameter_list['log_freq'] = 1                          #Logging frequence for console output
parameter_list['summery_freq'] = 1                      #Logging frequence for summeries
parameter_list['log_dir'] = '/summeries'               #Log directory for tensorboard summary
parameter_list['epochs'] = args.epoch                         #Number of epochs

flag = args.t
multi_flag = args.gpu

for i in parameter_list['experiments']:

    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/exp_' + str(i)
    parameter_list['checkpoint_expdir'] = parameter_list['checkpoint_dir']
    parameter_list['log_dir'] = parameter_list['checkpoint_dir'] + parameter_list['log_dir']
    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/checkpoints'

    pickle_name = parameter_list['checkpoint_dir'] + '/params.pickle'

    if not os.path.exists(parameter_list['checkpoint_dir']):
        os.makedirs(parameter_list['log_dir'])
        os.makedirs(parameter_list['checkpoint_dir'])
        os.makedirs(parameter_list['checkpoint_expdir'] + '/media')

        parameter_list['Experiment_No'] = i
        parameter_list['en_width'] = 3
        parameter_list['de_width'] = 3

        parameter_list['en_units'] = 80
        parameter_list['de_units'] = 80

        parameter_list['kaux_width'] = 2
        parameter_list['kaux_units'] = 40

    elif os.path.isfile(pickle_name):
        parameter_list = helpfunc.read_pickle(pickle_name)

    else:
        print('No pickle file exits at {}'.format(pickle_name))
        shutil.rmtree(parameter_list['checkpoint_expdir'])
        sys.exit()

    start = time.time()
    if multi_flag:
        if flag == 'train':
            print('Multi GPU {}ing'.format(flag))
            parameter_list['delta_t'] = args.delta_t
            parameter_list['learning_rate'] = parameter_list['learning_rate'] / len(tf.config.experimental.list_physical_devices('GPU'))
            parameter_list =  multi_train.traintest(copy.deepcopy(parameter_list))
        else:
            print('Testing...')
            parameter_list['delta_t'] = args.delta_t
            parameter_list =  testing.traintest(copy.deepcopy(parameter_list))
    else:
        print('Single/No GPU Training not available')
        

    if flag == 'train':
        helpfunc.write_pickle(parameter_list, pickle_name)
