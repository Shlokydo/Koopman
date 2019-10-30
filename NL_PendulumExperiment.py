import copy
import random
import numpy as np 
import pandas as pd 
import random as r
import os
import sys
import pickle
import shutil

import training_test
import multigpu_training_test as multi_train
import Helperfunction as helpfunc 

parameter_list = {}

#Basic setting for all the experiments
parameter_list['key'] = 'nl_pendulum'
parameter_list['num_timesteps'] = 51   #Next version, need to read it from the dataset
parameter_list['num_validation_points'] = 2000
parameter_list['input_scaling'] = 1
parameter_list['delta_t'] = 0.2

parameter_list['num_real'] = 0                          #Number of real Koopman eigenvalues
parameter_list['num_complex_pairs'] = 1                 #Number of complex conjugate eigenvalues
parameter_list['num_evals'] = 2          

parameter_list['experiments'] = 1
parameter_list['checkpoint_expdir'] = './results_nlpendulum'
parameter_list['checkpoint_dir'] = parameter_list['checkpoint_expdir'] + '/deeper_reconst_koop'

#Getting the default parameter_list
#Settings related to dataset creation
parameter_list['Batch_size'] = 256                      #Batch size
parameter_list['Buffer_size'] = 5000                    #Buffer size for shuffle

#Encoder layer
parameter_list['en_units'] = 25                         #Number of neurons in the encoder lstm layer
parameter_list['en_width'] = 2                          #Number of lstm layers
parameter_list['en_initializer'] = 'glorot_uniform'     #Initializer of layers
parameter_list['en_activation'] = 'tanh'                #Activation of layers

#Koopman auxilary network
parameter_list['kaux_units'] = 5                                                     #Number of neurons in dense layers
parameter_list['kaux_width'] = 2                                                      #Number of dense layers
parameter_list['kaux_output_units_real'] = parameter_list['num_real']                 #Number of real outputs
parameter_list['kaux_output_units_complex'] = parameter_list['num_complex_pairs'] * 2 #Number of complex outputs

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
parameter_list['early_stop_patience'] = 500               #Patience in num of epochs for early stopping
parameter_list['mth_step'] = 40                         #mth step for which prediction needs to be made
parameter_list['mth_cal_patience'] = 1                  #number of epochs after which mth loss is calculated
parameter_list['mth_no_cal_epochs'] = 40                #Number of epochs for which mth loss is not calculated
parameter_list['recon_hp'] = 0.001
parameter_list['global_epoch'] = 0

#Settings related to saving checkpointing and logging
parameter_list['max_checkpoint_keep'] = 4               #Max number of checkpoints to keep
parameter_list['num_epochs_checkpoint'] = 2            #Num of epochs after which to create a checkpoint
parameter_list['log_freq'] = 4                          #Logging frequence for console output
parameter_list['summery_freq'] = 1                      #Logging frequence for summeries
parameter_list['log_dir'] = '/summeries'               #Log directory for tensorboard summary
parameter_list['epochs'] = 500                         #Number of epochs

flag = 'train'
multi_flag = 1

for i in range(parameter_list['experiments']):
    
    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/exp_' + str(i+1)
    parameter_list['log_dir'] = parameter_list['checkpoint_dir'] + parameter_list['log_dir']
    parameter_list['model_loc'] = parameter_list['checkpoint_dir'] + '/model.json'
    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/checkpoints'

    pickle_name = parameter_list['checkpoint_dir'] + '/params.pickle'

    if not os.path.exists(parameter_list['checkpoint_dir']):
        os.makedirs(parameter_list['log_dir'])
        os.makedirs(parameter_list['checkpoint_dir'])

        parameter_list['Experiment_No'] = i
        #width = r.randint(5)
        en_width = 4
        parameter_list['en_width'] = en_width
        de_width = 4
        parameter_list['de_width'] = de_width
        
        if en_width == 4:
            en_units = 150
        if en_width == 6:
            en_units = r.randint(40,80)

        parameter_list['en_units'] = en_units
        de_units = 70
        parameter_list['de_units'] = de_units

        parameter_list['kaux_width'] = 1
        parameter_list['kaux_units'] = 170

    elif os.path.isfile(pickle_name):
        parameter_list = helpfunc.read_pickle(pickle_name)

    else:
        print('No pickle file exits at {}'.format(pickle_name))
        shutil.rmtree(parameter_list['checkpoint_dir'])
        sys.exit()

    if multi_flag:
        print('Multi GPU {}ing'.format(flag))
        parameter_list['global_epoch'] = multi_train.traintest(copy.deepcopy(parameter_list), flag)
    else:
        print('Single or no GPU {}ing'.format(flag))
        parameter_list['global_epoch'] = training_test.traintest(copy.deepcopy(parameter_list), flag)

    helpfunc.write_pickle(parameter_list, pickle_name)