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
parser.add_argument("--epoch", default=200, type=int, help="Set the number of epochs.")
parser.add_argument("--experiment", "--exp", default="d", help="Name of the experiment(s)")

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

study = optuna.create_study(direction = 'minimize', study_name='trial', storage='sqlite:///example.db', load_if_exists=True, pruner=optuna.pruners.PercentilePruner(80.0))
num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))

def get_params(trial):

    pl = {}

    #Basic setting for all the experiments
    pl['key'] = args.key
    pl['key_test'] = args.key + '_test'
    pl['num_timesteps'] = 51   #Next version, need to read it from the dataset
    pl['num_training_points'] = args.num_trainpoints
    pl['num_validation_points'] = args.num_valpoints
    pl['input_scaling'] = 1
    pl['inputs'] = 2
    pl['dataset'] = args.dataset + '.h5'
    pl['delta_t'] = args.delta_t

    pl['num_real'] = args.real_ef                          #Number of real Koopman eigenvalues
    pl['num_complex_pairs'] = args.complex_ef                 #Number of complex conjugate eigenvalues
    pl['num_evals'] = pl['num_real'] + 2 * pl['num_complex_pairs']

    pl['checkpoint_expdir'] = './optuna_' + '{}'.format(args.key)
    pl['checkpoint_dir'] = pl['checkpoint_expdir']

    #Getting the default pl
    #Settings related to dataset creation
    pl['Batch_size'] = int(1024 * num_gpu) if num_gpu else 1024                      #Batch size
    pl['Batch_size_val'] = int(1024 * num_gpu) if num_gpu else 1024                     #Batch size

    pl['Buffer_size'] = 50000                    #Buffer size for shuffle

    #Encoder layer
    pl['en_units_r'] = args.ende_units                         #Number of neurons in the encoder lstm layer
    pl['en_width_r'] = args.num_ende_layers                          #Number of lstm layers
    pl['en_activation'] = 'relu'                #All same as encoder till here
    pl['en_initializer'] = 'glorot_uniform'

    #Koopman auxilary network
    pl['kaux_units_real_r'] = args.kaux_units_real                                        
    pl['kaux_width_real_r'] = args.num_kaux_layers_real
    pl['kaux_units_complex_r'] = args.kaux_units_complex 
    pl['kaux_width_complex_r'] = args.num_kaux_layers_complex                                                      #Number of dense layers
    pl['kaux_output_units_real'] = pl['num_real']                 #Number of real outputs
    pl['kaux_output_units_complex'] = pl['num_complex_pairs'] * 2 #Number of complex outputs
    pl['kp_initializer'] = 'glorot_uniform'     #Initializer of layers
    pl['kp_activation'] = 'tanh'                #Activation of layers

    #Decoder layer
    pl['de_initializer'] = 'glorot_uniform'
    pl['de_activation'] = 'relu'                #All same as encoder till here
    pl['de_output_units'] = 2                   #Number of final output units

    pl['l_decay_param'] = 0.98

    #Settings related to trainig
    pl['learning_rate'] = trial.suggest_uniform('l_rate', 0.5e-3, 5e-3)                 #Initial learning rate
    pl['lr_decay_rate'] = args.lr_decayrate
    pl['learning_rate'] = pl['learning_rate'] * pl['Batch_size'] / 128.0
    pl['dropout'] = 0.0                         #Dropout for the layers
    pl['early_stop_patience'] =21900               #Patience in num of epochs for early stopping
    pl['mth_step'] = 40                         #mth step for which prediction needs to be made
    pl['mth_cal_patience'] = args.num_m_cal                  #number of epochs for which mth loss is calculated
    pl['mth_no_cal_epochs'] = args.num_m_no_cal                #Number of epochs for which mth loss is not calculated
    pl['only_RNN'] = 1500
    pl['weighted'] = args.weighted_loss
    pl['reconst_hp'] = 0.001
    pl['global_epoch'] = 0
    pl['val_min'] = 100

    #Settings related to saving checkpointing and logging
    pl['max_checkpoint_keep'] = 4               #Max number of checkpoints to keep
    pl['num_epochs_checkpoint'] = 2            #Num of epochs after which to create a checkpoint
    pl['log_freq'] = 1                          #Logging frequence for console output
    pl['summery_freq'] = 1                      #Logging frequence for summeries
    pl['log_dir'] = '/summeries'               #Log directory for tensorboard summary
    pl['epochs'] = args.epoch                         #Number of epochs
    pl['lr_decay_steps'] = int(pl['epochs'] * math.log(pl['lr_decay_rate']) / math.log(1.9/4))                  #No of steps for learning rate decay scheduling

    pl['checkpoint_dir'] = pl['checkpoint_dir'] + '/exp_' + args.experiment
    pl['checkpoint_expdir'] = pl['checkpoint_dir']
    pl['checkpoint_dir'] = pl['checkpoint_dir'] + '/checkpoints'
    
    pl['pickle_name'] = pl['checkpoint_expdir'] + '/optuna_params.pickle'

    try:
        pl['learning_rate'] = pl['learning_rate'] / num_gpu
    except:
        pass

    if args.t == 'train':

        if not os.path.exists(pl['checkpoint_expdir']):
            print('\nWill make a experiment directory\n')

        print('Multi GPU {}ing'.format(args.t))
        pl['checkpoint_dir'] = pl['checkpoint_dir'] + '_optuna'

    elif args.t == 'best':
        
        pl['checkpoint_dir'] = pl['checkpoint_dir'] + '_optbest'
        cd_copy = pl['checkpoint_dir']
        pl['log_dir'] = pl['checkpoint_dir'] + pl['log_dir']

        if not os.path.exists(pl['checkpoint_dir']):
            os.makedirs(pl['checkpoint_dir'] + '/media')
            os.makedirs(pl['log_dir'])
            pl['checkpoint_dir'] = pl['checkpoint_dir'] + '/checkpoints'
            os.makedirs(pl['checkpoint_dir'])

        print('Multi GPU {}ing'.format(args.t + ' param training'))
        pl['delta_t'] = args.delta_t

        pl['pickle_name'] = pl['checkpoint_dir'] + '/best_params.pickle'
        pl['global_epoch'] = 0
        pl['val_min'] = 100
    
    else:

        pl['checkpoint_dir'] = pl['checkpoint_dir'] + '_optbest'
        pl['pickle_name'] = pl['checkpoint_dir'] + '/best_params.pickle'

        if os.path.isfile(pl['pickle_name']):
            pl = helpfunc.read_pickle(pl['pickle_name'])
            print('Testing...')
            pl['delta_t'] = args.delta_t
            pl = testing.traintest(copy.deepcopy(pl))
        
        else:
            print('No pickle file exits at {}'.format(pl['pickle_name']))
            shutil.rmtree(pl['checkpoint_dir'])
            sys.exit()
    
    return pl

def get_optuna_param(trial, pl):

    pl['en_width'] = trial.suggest_int('num_en/de_layers', pl['en_width_r'][0], pl['en_width_r'][1])
    pl['de_width'] = pl['en_width']

    pl['en_units'] = []
    for i in range(pl['en_width']):
        pl['en_units'].append(trial.suggest_int('layer_' + str(i), pl['en_units_r'][0], pl['en_units_r'][1]))
    pl['de_units'] = pl['en_units'][::-1]

    pl['kaux_width_real'] = 0
    pl['kaux_units_real'] = []
    if pl['num_real']:
        pl['kaux_width_real'] = trial.suggest_int('num_kr_layers', pl['kaux_width_real_r'][0], pl['kaux_width_real_r'][1])
        for i in range(pl['kaux_width_real']):
            pl['kaux_units_real'].append(trial.suggest_int('kr_layer_' + str(i), pl['kaux_units_real_r'][0], pl['kaux_units_real_r'][1]))
        pl['kaux_units_real'].append(1)

    pl['kaux_width_complex'] = 0
    pl['kaux_units_complex'] = []
    if pl['num_complex_pairs']:
        pl['kaux_width_complex'] = trial.suggest_int('num_kc_layers', pl['kaux_width_complex_r'][0], pl['kaux_width_complex_r'][1])
        for i in range(pl['kaux_width_complex']):
            pl['kaux_units_complex'].append(trial.suggest_int('kc_layer_' + str(i), pl['kaux_units_complex_r'][0], pl['kaux_units_complex_r'][1]))
        pl['kaux_units_complex'].append(2)

    pl['checkpoint_dir'] = pl['checkpoint_dir'] + '/' + str(pl['en_width']) + str(pl['kaux_width_real']) + str(pl['kaux_width_complex']) + '_' + '_'.join(map(str, pl['en_units'])) + '_' + '_'.join(map(str, pl['kaux_units_complex']))
    pl['log_dir'] = pl['checkpoint_dir'] + pl['log_dir']
    if not os.path.exists(pl['checkpoint_dir']):
        os.makedirs(pl['log_dir'])

    return pl

def objective(trial):

    pl = get_params(trial)
    pl = get_optuna_param(trial, pl)
    return multi_train.traintest(trial, copy.deepcopy(pl))

if __name__ == "__main__":

    if args.t == 'train':
        study.optimize(objective, n_trials=args.num_optuna_trials)
        df = study.trials_dataframe()
        df.to_csv('optuna.csv')

    if args.t == 'best':
        best_case = optuna.trial.FixedTrial(study.best_params)
        objective(best_case)