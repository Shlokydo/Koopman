import tensorflow as tf
import math
import time
import numpy as np
import sys
import os

import network_arch as net
import Helperfunction as helpfunc
from plot import plot_figure, plot_diff, animate

def test(parameter_list, encoder, decoder, k_aux, k_jor, time_steps = 40, N_traj = 20):

    time, x_t_true = helpfunc.nl_pendulum(N= N_traj)
    prediction_list_global = []

    for j in range(x_t_true.shape[0]):
        prediction_list_local = []
        input_value = helpfunc.input_generator(x_t_true[j,0,:])
        prediction_list_local.append(input_value[0,0,:])
        
        input_value = encoder(input_value)
        for i in range(time_steps):
            koopman_embedding = k_aux(input_value)
            koopman_evolved = k_jor(koopman_embedding)
            prediction = decoder(koopman_evolved)
            input_value = prediction
            prediction_list_local.append(prediction.numpy()[0,0,:])
        
        k_aux.reset_states()
        x_t_local = np.asarray(prediction_list_local)
        prediction_list_global.append(prediction_list_local)

    x_t = np.asarray(prediction_list_global)
    x_diff = helpfunc.difference(x_t_true[:,2:time_steps+1,:], x_t[:,2:,:])
    plot_diff(x_diff[:,:,0], time, True, parameter_list['checkpoint_expdir']+'/media/x_variable.png')
    plot_diff(x_diff[:,:,1], time, True, parameter_list['checkpoint_expdir']+'/media/y_variable.png')
    x_t_true = np.concatenate((x_t_true[:,:time_steps+1,:], x_t), axis=0)
    plot_figure(x_t_true, True, parameter_list['checkpoint_expdir'] + '/media/nl_pendulum.png')
    animate(x_t_true, parameter_list['checkpoint_expdir'] + '/media/video.mp4')
    return None

def traintest(parameter_list, flag):

    print('\nGPU Available for testing: {}\n'.format(tf.test.is_gpu_available()))

    #Get the Model
    encoder = net.encoder(parameter_list = parameter_list)
    decoder = net.decoder(parameter_list = parameter_list)
    koopman_aux_net = net.koopman_aux_net(parameter_list = parameter_list)
    koopman_jordan = net.koopman_jordan(parameter_list = parameter_list)

    #Defining the checkpoint instance
    checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), encoder = encoder, decoder = decoder, koopman_aux_net = koopman_aux_net, koopman_jordan = koopman_jordan)

    #Creating checkpoint instance
    save_directory = parameter_list['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory,
                                        max_to_keep= parameter_list['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        print('Starting testing \n')
        return test(parameter_list, encoder, decoder, koopman_aux_net, koopman_jordan)

    else:
        print("No checkpoint exists. Quiting.....")
        sys.exit()

    print(learning_rate)