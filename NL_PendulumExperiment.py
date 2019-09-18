import copy
import random
import numpy as np 
import pandas as pd 
import random as r
import os
import sys

import params
import training_test
import Helperfunction as helpfunc 

#Getting the default parameter_list
parameter_list = params.parameter_list

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

parameter_list['learning_rate'] = parameter_list['learning_rate'] * parameter_list['Batch_size'] / 256.0
parameter_list['mth_step'] = 40                         #mth step for which prediction needs to be made
parameter_list['mth_cal_patience'] = 1                  #Patience for calculation of mth step

parameter_list['checkpoint_dir'] = './deeper_reconst_koop'
parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/nl_pendulum'

flag = 'test'

for i in range(parameter_list['experiments']):
    
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
    
    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/exp_' + str(i+1)
    parameter_list['log_dir'] = parameter_list['checkpoint_dir'] + '/summeries'
    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/checkpoints'
    

    if not os.path.exists(parameter_list['checkpoint_dir']):
        os.makedirs(parameter_list['log_dir'])
        os.makedirs(parameter_list['checkpoint_dir'])

    csv_name = parameter_list['checkpoint_dir'] + '/params.csv'

    if os.path.isfile(csv_name):
        parameter_list = helpfunc.read_dataframe(csv_name)
    
    parameter_list = training_test.traintest(copy.deepcopy(parameter_list), flag)
    params_dataframe = pd.DataFrame(parameter_list, index=[1])

    helpfunc.write_dataframe(params_dataframe, csv_name)