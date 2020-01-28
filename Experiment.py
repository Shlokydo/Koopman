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
import math

import optuna

import multigpu_training_test as multi_train
import Helperfunction as helpfunc
import tensorflow as tf
import valtest as testing

parser = argparse.ArgumentParser(description='Experiment controller')
parser.add_argument("--t", default='train', type=str, help="Training or testing flag setter", choices=["train", "best", "test"])
parser.add_argument("--gpu", default=1, type=int, help="To enable/disable multi-gpu training. Default, true.", choices=[0, 1])
parser.add_argument("--epoch", default=200, type=int, help="Set the number of epochs.")
parser.add_argument("--experiment", "--exp", default="d", help="Name of the experiment(s)", nargs='*')

parser.add_argument("--key", default="nl_pendulum", help="Key for the dataset to be used from the HDF5 file")
parser.add_argument("--dataset", default="Dataset", help="Name of the .h5 dataset to be used for training")
parser.add_argument("--delta_t", default=0.2, type=float, help="Time stepping")
parser.add_argument("--num_ts", default=51, type=int, help="num of time steps")
parser.add_argument("--num_valpoints", "--n_vals", default=4096, type=int, help="num of validation trajectories")
parser.add_argument("--num_trainpoints", "--n_train", default=8192, type=int, help="num of training trajectories")

parser.add_argument("--num_ende_layers", "--n_ende", default =[2, 3], type=int, nargs="+", help="Lower and upper limit for num of encoder/decoder layers")
parser.add_argument("--ende_units", "--u_ende", default = [70, 100], type=int, nargs="+", help="Lower and upper limit for num of encoder/decoder units")
parser.add_argument("--num_kaux_layers_real", "--rn_kaux", default = [1, 3], type=int, nargs="+", help="Lower and upper limit for num of real Koopman aux layers")
parser.add_argument("--num_kaux_layers_complex", "--cn_kaux", default = [1, 3], type=int, nargs="+", help="Lower and upper limit for num of complex Koopman aux layers")
parser.add_argument("--kaux_units_real", "--ru_kaux", default=[100, 200], type=int, nargs="+", help="Lower and upper limit for num of real Koopman aux units per layer")
parser.add_argument("--kaux_units_complex", "--cu_kaux", default=[100, 200], type=int, nargs="+", help="Lower and upper limit for num of complex Koopman aux units per layer")

parser.add_argument("--real_ef", default=1, type=int, help="Number of real eigenfunctions")
parser.add_argument("--complex_ef", default=1, type=int, help="Number of complex eigenfunctions")
parser.add_argument("--num_optuna_trials", "--opt_trials", type=int, default=10, help='num of Optuna study trials to make')

parser.add_argument("--weighted_loss", "--wl", default=0, type=int, help="Weighted loss function as in Otto et.al")
parser.add_argument("--lr_decayrate", "--lrdr", default=0.1, type=float, help="Learning rate decay rate")

parser.add_argument("--num_m_no_cal", "--m_no", default=40, type=int, help="Num of epoch without mth calculation")
parser.add_argument("--num_m_cal", "--m_cal", default=10, type=int, help="Num of epoch with mth calculation")
args = parser.parse_args()

def get_params(trial):

    parameter_list = {}

    #Basic setting for all the experiments
    parameter_list['key'] = args.key
    parameter_list['key_test'] = args.key + '_test'
    parameter_list['num_timesteps'] = 51   #Next version, need to read it from the dataset
    parameter_list['num_training_points'] = args.num_trainpoints
    parameter_list['num_validation_points'] = args.num_valpoints
    parameter_list['input_scaling'] = 1
    parameter_list['inputs'] = 2
    parameter_list['dataset'] = args.dataset + '.h5'

    parameter_list['num_real'] = args.real_ef                          #Number of real Koopman eigenvalues
    parameter_list['num_complex_pairs'] = args.complex_ef                 #Number of complex conjugate eigenvalues
    parameter_list['num_evals'] = parameter_list['num_real'] + 2 * parameter_list['num_complex_pairs']

    parameter_list['experiments'] = args.experiment
    parameter_list['checkpoint_expdir'] = './optuna_' + '{}'.format(args.key)
    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_expdir']

    #Getting the default parameter_list
    #Settings related to dataset creation
    parameter_list['Batch_size'] = int(1024 * len(tf.config.experimental.list_physical_devices('GPU')))                      #Batch size
    parameter_list['Batch_size_val'] = int(1024 * len(tf.config.experimental.list_physical_devices('GPU')))                    #Batch size
    parameter_list['Buffer_size'] = 50000                    #Buffer size for shuffle

    #Encoder layer
    parameter_list['en_units_r'] = args.ende_units                         #Number of neurons in the encoder lstm layer
    parameter_list['en_width_r'] = args.num_ende_layers                          #Number of lstm layers
    parameter_list['en_activation'] = 'relu'                #All same as encoder till here
    parameter_list['en_initializer'] = 'glorot_uniform'

    #Koopman auxilary network
    parameter_list['kaux_units_real_r'] = args.kaux_units_real                                        
    parameter_list['kaux_width_real_r'] = args.num_kaux_layers_real
    parameter_list['kaux_units_complex_r'] = args.kaux_units_complex 
    parameter_list['kaux_width_complex_r'] = args.num_kaux_layers_complex                                                      #Number of dense layers
    parameter_list['kaux_output_units_real'] = parameter_list['num_real']                 #Number of real outputs
    parameter_list['kaux_output_units_complex'] = parameter_list['num_complex_pairs'] * 2 #Number of complex outputs
    parameter_list['kp_initializer'] = 'glorot_uniform'     #Initializer of layers
    parameter_list['kp_activation'] = 'tanh'                #Activation of layers

    #Decoder layer
    parameter_list['de_initializer'] = 'glorot_uniform'
    parameter_list['de_activation'] = 'relu'                #All same as encoder till here
    parameter_list['de_output_units'] = 2                   #Number of final output units

    parameter_list['l_decay_param'] = 0.98

    #Settings related to trainig
    parameter_list['learning_rate'] = 0.001                 #Initial learning rate
    parameter_list['lr_decay_rate'] = args.lr_decayrate
    parameter_list['learning_rate'] = parameter_list['learning_rate'] * parameter_list['Batch_size'] / 128.0
    parameter_list['dropout'] = 0.0                         #Dropout for the layers
    parameter_list['early_stop_patience'] =21900               #Patience in num of epochs for early stopping
    parameter_list['mth_step'] = 40                         #mth step for which prediction needs to be made
    parameter_list['mth_cal_patience'] = args.num_m_cal                  #number of epochs for which mth loss is calculated
    parameter_list['mth_no_cal_epochs'] = args.num_m_no_cal                #Number of epochs for which mth loss is not calculated
    parameter_list['only_RNN'] = 1500
    parameter_list['weighted'] = args.weighted_loss
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
    #parameter_list['lr_decay_steps'] = parameter_list['epochs'] * parameter_list['num_training_points'] / parameter_list['Batch_size']                  #No of steps for learning rate decay scheduling
    parameter_list['lr_decay_steps'] = int(parameter_list['epochs'] * math.log(parameter_list['lr_decay_rate']) / math.log(1.9/4))                  #No of steps for learning rate decay scheduling

    flag = args.t
    multi_flag = args.gpu

    for i in parameter_list['experiments']:

        parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/exp_' + str(i)
        parameter_list['checkpoint_expdir'] = parameter_list['checkpoint_dir']
        parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/checkpoints'
        
        study = optuna.create_study(direction = 'minimize', study_name='trial', storage='sqlite:///example.db', load_if_exists=True, pruner=optuna.pruners.PercentilePruner(25.0))

        parameter_list['pickle_name'] = parameter_list['checkpoint_expdir'] + '/optuna_params.pickle'

        if flag == 'train':

            if not os.path.exists(parameter_list['checkpoint_expdir']):
                print('\nWill make a experiment directory\n')

            elif os.path.isfile(parameter_list['pickle_name']):
                parameter_list = helpfunc.read_pickle(parameter_list['pickle_name'])
            
            else:
                print('No pickle file exits at {}'.format(parameter_list['pickle_name']))
                shutil.rmtree(parameter_list['checkpoint_expdir'])
                sys.exit()

            print('Multi GPU {}ing'.format(flag))
            parameter_list['delta_t'] = args.delta_t
            parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '_optuna'

            parameter_list['learning_rate'] = parameter_list['learning_rate'] / len(tf.config.experimental.list_physical_devices('GPU'))
            study.optimize(lambda trial: multi_train.traintest(trial, copy.deepcopy(parameter_list), flag), n_trials=args.num_optuna_trials)
            df = study.trials_dataframe()
            df.to_csv('optuna.csv')

        elif flag == 'best':

            if os.path.isfile(parameter_list['pickle_name']):
                parameter_list = helpfunc.read_pickle(parameter_list['pickle_name'])
            
            else:
                print('No pickle file exits at {}'.format(parameter_list['pickle_name']))
                shutil.rmtree(cd_copy)
                sys.exit()
            
            parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '_optbest'
            cd_copy = parameter_list['checkpoint_dir']
            parameter_list['log_dir'] = parameter_list['checkpoint_dir'] + parameter_list['log_dir']

            if not os.path.exists(parameter_list['checkpoint_dir']):
                os.makedirs(parameter_list['checkpoint_dir'] + '/media')
                os.makedirs(parameter_list['log_dir'])
                parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/checkpoints'
                os.makedirs(parameter_list['checkpoint_dir'])

            print('Multi GPU {}ing'.format(flag + ' param training'))
            parameter_list['delta_t'] = args.delta_t
            parameter_list['learning_rate'] = parameter_list['learning_rate'] / len(tf.config.experimental.list_physical_devices('GPU'))
            best_case = optuna.trial.FixedTrial(study.best_params)

            parameter_list['pickle_name'] = parameter_list['checkpoint_dir'] + '/best_params.pickle'
            parameter_list['global_epoch'] = 0
            parameter_list['val_min'] = 100
            multi_train.traintest(best_case, copy.deepcopy(parameter_list), flag)
        
        else:

            parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '_optbest'
            parameter_list['pickle_name'] = parameter_list['checkpoint_dir'] + '/best_params.pickle'

            if os.path.isfile(parameter_list['pickle_name']):
                parameter_list = helpfunc.read_pickle(parameter_list['pickle_name'])
                print('Testing...')
                parameter_list['delta_t'] = args.delta_t
                parameter_list = testing.traintest(copy.deepcopy(parameter_list))
            
            else:
                print('No pickle file exits at {}'.format(parameter_list['pickle_name']))
                shutil.rmtree(parameter_list['checkpoint_dir'])
                sys.exit()


def get_optuna_param(trial, parameter_list):

    parameter_list['en_width'] = trial.suggest_int('num_en/de_layers', parameter_list['en_width_r'][0], parameter_list['en_width_r'][1])
    parameter_list['de_width'] = parameter_list['en_width']

    parameter_list['en_units'] = []
    for i in range(parameter_list['en_width']):
        parameter_list['en_units'].append(trial.suggest_int('layer_' + str(i), parameter_list['en_units_r'][0], parameter_list['en_units_r'][1]))
    parameter_list['de_units'] = parameter_list['en_units'][::-1]

    parameter_list['kaux_width_real'] = 0
    parameter_list['kaux_units_real'] = []
    if parameter_list['num_real']:
        parameter_list['kaux_width_real'] = trial.suggest_int('num_kr_layers', parameter_list['kaux_width_real_r'][0], parameter_list['kaux_width_real_r'][1])
        for i in range(parameter_list['kaux_width_real']):
            parameter_list['kaux_units_real'].append(trial.suggest_int('kr_layer_' + str(i), parameter_list['kaux_units_real_r'][0], parameter_list['kaux_units_real_r'][1]))
        parameter_list['kaux_units_real'].append(1)

    parameter_list['kaux_width_complex'] = 0
    parameter_list['kaux_units_complex'] = []
    if parameter_list['num_complex_pairs']:
        parameter_list['kaux_width_complex'] = trial.suggest_int('num_kc_layers', parameter_list['kaux_width_complex_r'][0], parameter_list['kaux_width_complex_r'][1])
        for i in range(parameter_list['kaux_width_complex']):
            parameter_list['kaux_units_complex'].append(trial.suggest_int('kc_layer_' + str(i), parameter_list['kaux_units_complex_r'][0], parameter_list['kaux_units_complex_r'][1]))
        parameter_list['kaux_units_complex'].append(2)

    parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + '/' + str(parameter_list['en_width']) + str(parameter_list['kaux_width_real']) + str(parameter_list['kaux_width_complex']) + '_' + '_'.join(map(str, parameter_list['en_units'])) + '_' + '_'.join(map(str, parameter_list['kaux_units_complex']))
    parameter_list['log_dir'] = parameter_list['checkpoint_dir'] + parameter_list['log_dir']
    if not os.path.exists(parameter_list['checkpoint_dir']):
        os.makedirs(parameter_list['log_dir'])

    return parameter_list

def objective(trial, parameter_list):

    parameter_list = get_optuna_param(trial, parameter_list)
    return multi_train.traintest(trial, copy.deepcopy(parameter_list))
