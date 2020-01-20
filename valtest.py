import tensorflow as tf
import math
import time
import numpy as np
import sys
import os

import network_arch as net
import Helperfunction as helpfunc
from plot import plot_figure, plot_diff, animate

def test(parameter_list, encoder, decoder, k_aux, k_jor, time_steps = 30, N_traj = 20, seconds = 2):

    time = np.arange(0, seconds, parameter_list['delta_t'])
    steps = len(time)
    dataframe = helpfunc.import_datset(parameter_list['dataset'], parameter_list['key_test'])

    #Converting dataframe to numpy array
    nparray = helpfunc.dataframe_to_nparray(dataframe)

    #Delete the unrequired columns in the nparray
    nparray = helpfunc.nparray_del_vec_along_axis(nparray, 0, ax = 1)

    #Change the datatype to float32
    initial_dataset = helpfunc.change_nparray_datatype(nparray, 'float32')

    #Scale the dataset
    initial_dataset = helpfunc.dataset_scaling(initial_dataset, parameter_list['input_scaling'])

    #Generate split x and y sequence from the dataset
    initial_dataset_x = helpfunc.sequences_test(initial_dataset, parameter_list['num_timesteps'])

    x_t_true = initial_dataset_x[:N_traj]
    extension_list = x_t_true[:,-1,:]

    x_t = helpfunc.nl_pendulum(extension_list, time = time[: int(steps - parameter_list['num_timesteps'] + 1)])
    x_t_true = np.concatenate((x_t_true, x_t[:,1:,:]), axis=1)
    prediction_list_global = []
    k_embeddings_list_global = []
    eigen_value_global = []

    for j in range(x_t_true.shape[0]):
        prediction_list_local = []
        k_embeddings_list_local = []
        eigen_value_local = []
        input_value = helpfunc.input_generator(x_t_true[j,0,:])
        prediction_list_local.append(input_value[0,0,:])
        
        #Createing first encoded state of all the trajectories
        k_embeddings_cur = encoder(input_value)

        #Appending embedding
        k_embeddings_list_local.append(k_embeddings_cur.numpy()[0,0,:])
        for i in range(steps-1):

            #Finding the associated eigenvalues
            k_omegas = k_aux(k_embeddings_cur)
            
            #Preparing input for the Koopam Jordan
            k_jordan_input = tf.concat([k_omegas, k_embeddings_cur], axis= 2)

            #Evolving the embedding
            koopman_evolved = k_jor(k_jordan_input)
            
            #Appending the evolved embedding
            k_embeddings_list_local.append(koopman_evolved.numpy()[0,0,:])
            
            #Decoding the embedding
            prediction = decoder(koopman_evolved)

            #Reseting the evolved embedding to current step embedding
            k_embeddings_cur = koopman_evolved

            #Appending the eigenvalues and prediction 
            eigen_value_local.append(k_omegas.numpy()[0,0,:])
            prediction_list_local.append(prediction.numpy()[0,0,:])     
        
        k_aux.reset_states()
        prediction_list_global.append(prediction_list_local)
        k_embeddings_list_global.append(k_embeddings_list_local)
        eigen_value_global.append(eigen_value_local)

    x_t = np.asarray(prediction_list_global)
    k_embeddings = np.asarray(k_embeddings_list_global)
    eigen_value = np.asarray(eigen_value_global)
    x_diff = helpfunc.difference(x_t_true, x_t)
    plot_diff(x_diff[:,:,0], time, True, parameter_list['checkpoint_expdir']+'/media/x_variable.png', title = 'X error')
    plot_diff(x_diff[:,:,1], time, True, parameter_list['checkpoint_expdir']+'/media/y_variable.png', title = 'Y error')
    x_t_true = np.concatenate((x_t_true, x_t), axis=0)
    plot_figure(x_t_true, True, parameter_list['checkpoint_expdir'] + '/media/nl_pendulum.png', title = 'Trajectory evolution')
    plot_figure(k_embeddings, True, parameter_list['checkpoint_expdir'] + '/media/k_embeddings.png', statespace=False, embed=True, title = 'Koopman Embeddings')
    plot_figure(eigen_value, True, parameter_list['checkpoint_expdir'] + '/media/eigen_value.png', statespace=False, evalue=True, title = 'Eigenvalue scatter plot')
    animate(x_t_true, parameter_list['checkpoint_expdir'] + '/media/video.mp4')
    return None

def traintest(parameter_list):

    print('\nGPU Available for testing: {}\n'.format(tf.test.is_gpu_available()))

    parameter_list['stateful'] = True

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
