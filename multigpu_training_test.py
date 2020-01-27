import tensorflow as tf
import math
import time
import numpy as np
import sys
import os

import network_arch as net
import Helperfunction as helpfunc
from plot import plot_figure, plot_diff, animate

tf.autograph.set_verbosity(10)

mirrored_strategy = tf.distribute.MirroredStrategy()

def train(parameter_list, preliminary_net, loop_net, checkpoint, manager, summary_writer, optimizer):
    #Importing dataset into dataframe
    dataframe = helpfunc.import_datset(parameter_list['dataset'], parameter_list['key'])

    #Converting dataframe to numpy array
    nparray = helpfunc.dataframe_to_nparray(dataframe)

    #Delete the unrequired columns in the nparray
    nparray = helpfunc.nparray_del_vec_along_axis(nparray, 0, ax = 1)

    #Change the datatype to float32
    initial_dataset = helpfunc.change_nparray_datatype(nparray, 'float32')

    #Scale the dataset
    initial_dataset = helpfunc.dataset_scaling(initial_dataset, parameter_list['input_scaling'])

    #Generate split x and y sequence from the dataset
    initial_dataset_x, initial_dataset_y = helpfunc.split_sequences(initial_dataset, parameter_list['num_timesteps'])

    #Shuffling the dataset
    initial_dataset_x, initial_dataset_y = helpfunc.np_array_shuffle(initial_dataset_x, initial_dataset_y)

    #Splitting the dataset into train and validation dataset
    initial_dataset_x = initial_dataset_x[:(parameter_list['num_validation_points']+parameter_list['num_training_points'])]
    initial_dataset_y = initial_dataset_y[:(parameter_list['num_validation_points']+parameter_list['num_training_points'])]

    initial_dataset_x_train = initial_dataset_x[:-parameter_list['num_validation_points']]
    initial_dataset_y_train = initial_dataset_y[:-parameter_list['num_validation_points']]

    initial_dataset_x_val = initial_dataset_x[-parameter_list['num_validation_points']:]
    initial_dataset_y_val = initial_dataset_y[-parameter_list['num_validation_points']:]

    #Creating tensorflow dataset for train and val datasets
    tf_dataset_x = helpfunc.create_tfdataset(initial_dataset_x_train)
    tf_dataset_y = helpfunc.create_tfdataset(initial_dataset_y_train)

    tf_dataset_x_val = helpfunc.create_tfdataset(initial_dataset_x_val)
    tf_dataset_y_val = helpfunc.create_tfdataset(initial_dataset_y_val)

    #Batching the dataset
    tf_dataset_batch_x = tf_dataset_x.batch(batch_size= parameter_list['Batch_size'], drop_remainder= False)
    tf_dataset_batch_y = tf_dataset_y.batch(batch_size= parameter_list['Batch_size'], drop_remainder= False)

    tf_dataset_batch_x_val = tf_dataset_x_val.batch(batch_size= parameter_list['Batch_size_val'], drop_remainder= False)
    tf_dataset_batch_y_val = tf_dataset_y_val.batch(batch_size= parameter_list['Batch_size_val'], drop_remainder= False)

    #Zipping the dataset
    dataset = tf.data.Dataset.zip((tf_dataset_batch_x, tf_dataset_batch_y))
    val_dataset = tf.data.Dataset.zip((tf_dataset_batch_x_val, tf_dataset_batch_y_val))

    #Shuffling the dataset again (kinda redundant but no problem so why not)
    dataset = dataset.shuffle(parameter_list['Buffer_size'], reshuffle_each_iteration=True)

    #Distributing the dataset
    dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)

    decaying_weights = np.asarray([math.pow(parameter_list['l_decay_param'], j) for j in range(parameter_list['num_timesteps'] - 1)])
    decaying_weights_sum = np.sum(decaying_weights)
    decaying_weights = decaying_weights/decaying_weights_sum
    decaying_weights = tf.convert_to_tensor(decaying_weights, dtype = 'float32')

    with mirrored_strategy.scope():

        #loss function for training
        loss_func = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM, name='LossMSE')

        #Metric
        metric = tf.keras.metrics.RootMeanSquaredError(name='MetricRMSE')

        def compute_loss(labels, predictions, weighted):
            
            if weighted:
                sub = tf.subtract(labels, predictions)
                squ_trans_mul = tf.transpose((tf.transpose(tf.square(sub), perm = [0,2,1]) * decaying_weights[:labels.shape[1]]), perm=[0,2,1])
                per_example_loss = tf.reduce_sum(squ_trans_mul)
            else:
                per_example_loss = loss_func(labels, predictions)

            return per_example_loss / (parameter_list['Batch_size'] * (parameter_list['num_timesteps'] - 1))

        def compute_metric(labels, predictions):
            per_example_metric = metric(labels, predictions)
            return per_example_metric

        def train_step(inputs):
            
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            (t_current, t_next_actual), mth_flag = inputs

            with tf.GradientTape() as tape:
                t_next_predicted, t_reconstruction, t_embedding, t_jordan = preliminary_net(t_current)

                #Calculating relative loss
                loss_next_prediction = compute_loss(t_next_actual, t_next_predicted, weighted = 0) * 0.001
                reconstruction_loss = compute_loss(t_current, t_reconstruction, weighted = 0) * 0.001
                linearization_loss = compute_loss(t_embedding, t_jordan, weighted = 0) / parameter_list['num_timesteps'] 
                #linearization_loss = compute_loss(t_embedding, t_jordan, weighted = parameter_list['weighted'])  

            gradients = tape.gradient([loss_next_prediction, linearization_loss, reconstruction_loss], preliminary_net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, preliminary_net.trainable_weights))           

            if mth_flag:
                with tf.GradientTape() as tape:     
                    t_mth_predictions = loop_net(t_current)
                    loss_mth = loss_func(t_mth_predictions, t_next_actual[:, parameter_list['mth_step']:,:]) / (parameter_list['Batch_size'] * (parameter_list['num_timesteps'] - parameter_list['mth_step'] - 1)) * 0.005
                
                loop_net.layers[0].trainable=False
                loop_net.layers[3].trainable=False
                gradients = tape.gradient([loss_mth], loop_net.trainable_weights)
                optimizer.apply_gradients(zip(gradients, loop_net.trainable_weights))           
                loop_net.layers[0].trainable=True
                loop_net.layers[3].trainable=True
            else:
                loss_mth = tf.constant(0, dtype=tf.float32)

            metric_device = compute_metric(t_next_actual, t_next_predicted)

            return loss_next_prediction, metric_device, reconstruction_loss, linearization_loss, loss_mth

        def val_step(inputs):

            (t_current, t_next_actual), mth_flag = inputs

            t_next_predicted, t_reconstruction, t_embedding, t_jordan = preliminary_net(t_current)

            #Calculating relative loss
            loss_next_prediction = compute_loss(t_next_actual, t_next_predicted, 0) * 0.001
            reconstruction_loss = compute_loss(t_current, t_reconstruction, 0) * 0.001
            linearization_loss = compute_loss(t_embedding, t_jordan, 0) / parameter_list['num_timesteps'] 

            if mth_flag:
                t_mth_predictions = loop_net(t_current)
                loss_mth = loss_func(t_mth_predictions, t_next_actual[:, parameter_list['mth_step']:,:]) / (parameter_list['Batch_size'] * (parameter_list['num_timesteps'] - parameter_list['mth_step'] - 1))
            else:
                loss_mth = tf.constant(0, dtype=tf.float32)

            metric_device = compute_metric(t_next_actual, t_next_predicted)

            return loss_next_prediction, metric_device, reconstruction_loss, linearization_loss, loss_mth

        @tf.function
        def distributed_train(inputs):
            
            pr_losses, pr_metric, pr_reconst, pr_lin, pr_mth = mirrored_strategy.experimental_run_v2(train_step, args=(inputs,))

            losses = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_losses, axis=None)
            metric = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metric, axis=None)
            reconst = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_reconst, axis=None)
            lin = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_lin, axis=None)
            mth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_mth, axis = None)

            return losses, metric, reconst, lin, mth

        @tf.function
        def distributed_val(inputs):

            pr_losses, pr_metric, pr_reconst, pr_lin, pr_mth = mirrored_strategy.experimental_run_v2(val_step, args=(inputs,))

            losses = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_losses, axis=None)
            metric = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metric, axis=None)
            reconst = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_reconst, axis=None)
            lin = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_lin, axis=None)
            mth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_mth, axis = None)

            return losses, metric, reconst, lin, mth

        #Initialing training variables
        global_epoch = parameter_list['global_epoch']
        global_step = 0
        val_min = 0
        val_loss_min = parameter_list['val_min']
        timer_tot = time.time()

        cal_mth_loss_flag = False
        mth_loss_cal_mani = parameter_list['mth_no_cal_epochs']

        with summary_writer.as_default():

            epochs = parameter_list['epochs']

            for epoch in range(epochs):

                global_epoch += 1

                start_time = time.time()
                print('\nStart of epoch {}'.format(global_epoch))

                if (epoch > parameter_list['only_RNN']):
                    if not((global_epoch) % mth_loss_cal_mani):
                        if not(cal_mth_loss_flag):
                            print("\n Starting calculation of mth prediction loss \n")
                            cal_mth_loss_flag = True
                            mth_loss_cal_mani = parameter_list['mth_cal_patience']
                        else:
                            cal_mth_loss_flag = False
                            mth_loss_cal_mani = parameter_list['mth_no_cal_epochs']
                            print("Stopping mth calculation loss")

                # Iterate over the batches of the dataset.
                for step, inp in enumerate(dataset):

                    global_step += 1

                    inputs = (inp, cal_mth_loss_flag) 
                    loss, t_metric, t_reconst, t_lin, t_mth = distributed_train(inputs)

                    if not (step % parameter_list['log_freq']):
                        print('Training loss (for one batch) at step {}: {}'.format(step+1, float(loss)))
                        print('Seen so  far: {} samples'.format(global_step * parameter_list['Batch_size']))

                print('Training acc over epoch: {} \n'.format(float(t_metric)))

                if not (global_epoch % parameter_list['summery_freq']):
                    tf.summary.scalar('RootMSE error', t_metric, step= global_epoch)
                    tf.summary.scalar('Loss RNN', loss, step= global_epoch)
                    tf.summary.scalar('Reconstruction_loss', t_reconst, step= global_epoch)
                    tf.summary.scalar('Linearization_loss', t_lin, step= global_epoch)
                    tf.summary.scalar('Learning rate', optimizer._decayed_lr(var_dtype=tf.float32).numpy(), step= global_epoch)
                    if cal_mth_loss_flag:
                        tf.summary.scalar('Mth prediction loss', t_mth, step = global_epoch)

                for v_step, v_inp in enumerate(val_dataset):

                    v_inputs = (v_inp, cal_mth_loss_flag)
                    v_loss, v_metric, v_reconst, v_lin, v_mth = distributed_val(v_inputs)

                    if not (v_step % parameter_list['log_freq']):
                        print('Validation loss (for one batch) at step {} : {}'.format(v_step + 1, float(v_loss)))

                print('Validation acc over epoch: {} \n'.format(float(v_metric)))

                if not (global_epoch % parameter_list['summery_freq']):
                    tf.summary.scalar('V_RootMSE error', v_metric, step= global_epoch)
                    tf.summary.scalar('V_Loss RNN', v_loss, step= global_epoch)
                    tf.summary.scalar('V_Reconstruction_loss', v_reconst, step= global_epoch)
                    tf.summary.scalar('V_Linearization_loss', v_lin, step= global_epoch)
                    if cal_mth_loss_flag:
                        tf.summary.scalar('V_Mth prediction loss', v_mth, step = global_epoch)

                if (global_epoch > parameter_list['only_RNN']):
                    if val_loss_min > v_loss:
                        val_loss_min = v_loss
                        checkpoint.epoch.assign_add(1)
                        if  not (int(checkpoint.epoch + 1) % parameter_list['num_epochs_checkpoint']):
                            save_path = manager.save()
                            print("Saved checkpoint for epoch {}: {}".format(checkpoint.epoch.numpy(), save_path))
                            print("loss {}".format(loss.numpy()))

                if math.isnan(v_metric):
                    print('Breaking out as the validation loss is nan')
                    break

                #if (global_epoch > 19):
                #    if not (epoch % parameter_list['early_stop_patience']):
                #        if  not (val_min):
                #            val_min = v_metric
                #        else:
                #            if val_min > v_metric:
                #                val_min = v_metric
                #            else:
                #                print('Breaking loop  as validation accuracy not improving')
                #                print("loss {}".format(loss.numpy()))
                #                break

                print('Time for epoch (in seconds): %s' %((time.time() - start_time)))

    print('\n Total Epoch time (in minutes): {}'.format((time.time()-timer_tot)/60))
    parameter_list['global_epoch'] = global_epoch
    parameter_list['val_min'] = val_loss_min
    print('\nNumber of iterations for optimizer: {}'.format(optimizer.iterations.numpy()))

    helpfunc.write_pickle(parameter_list, parameter_list['pickle_name'])
    
    return val_loss_min

def traintest(trial, parameter_list, flag):

    print('\nGPU Available: {}\n'.format(tf.test.is_gpu_available()))

    #Get the Model
    with mirrored_strategy.scope():

        parameter_list, encoder, decoder, kaux_real, kaux_complex = get_model(trial, parameter_list, flag)
        koopman_aux_net = net.koopman_aux_net(kaux_real, kaux_complex, parameter_list = parameter_list)
        koopman_jordan = net.koopman_jordan(parameter_list = parameter_list)

        preliminary_net = net.preliminary_net(parameter_list, encoder, decoder, koopman_aux_net, koopman_jordan)
        loop_net = net.loop_net(parameter_list, encoder, decoder, koopman_aux_net, koopman_jordan)

        #Defining Model compiling parameters
        learningrate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(parameter_list['learning_rate'], decay_steps = parameter_list['lr_decay_steps'], decay_rate = parameter_list['lr_decay_rate'], staircase = False)
        #learning_rate = learningrate_schedule
        learning_rate = parameter_list['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #Defining the checkpoint instance
        checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), encoder = encoder, decoder = decoder, kaux_real = kaux_real, kaux_complex = kaux_complex, koopman_jordan = koopman_jordan, optimizer = optimizer)

    #Creating summary writer
    summary_writer = tf.summary.create_file_writer(logdir= parameter_list['log_dir'])

    #Creating checkpoint instance
    save_directory = parameter_list['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, max_to_keep= parameter_list['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        print('Starting training of Experiment... \n')
        return train(parameter_list, preliminary_net, loop_net, checkpoint, manager, summary_writer, optimizer)

    else:
        print("No checkpoint exists.")

        print('Initializing from scratch for Experiment... \n')
        return train(parameter_list, preliminary_net, loop_net, checkpoint, manager, summary_writer, optimizer)
<<<<<<< HEAD
=======

def get_model(trial, parameter_list, flag):

    parameter_list['en_width'] = trial.suggest_int('num_en/de_layers', 1, 4)
    parameter_list['de_width'] = parameter_list['en_width']

    if parameter_list['num_real']:
        parameter_list['kaux_width_real'] = trial.suggest_int('num_kr_layers', 1, 2)
    if parameter_list['num_complex_pairs']:
        parameter_list['kaux_width_complex'] = trial.suggest_int('num_kc_layers', 1, 2)

    parameter_list['en_units'] = []
    for i in range(parameter_list['en_width']):
        parameter_list['en_units'].append(trial.suggest_int('layer_' + str(i), 20, 150))
    parameter_list['de_units'] = parameter_list['en_units'][::-1]


    parameter_list['kaux_units_real'] = []
    for i in range(parameter_list['kaux_width_real']):
        parameter_list['kaux_units_real'].append(trial.suggest_int('kr_layer_' + str(i), 70, 200))
    parameter_list['kaux_units_real'].append(1)

    parameter_list['kaux_units_complex'] = []
    for i in range(parameter_list['kaux_width_complex']):
        parameter_list['kaux_units_complex'].append(trial.suggest_int('kc_layer_' + str(i), 70, 200))
    parameter_list['kaux_units_complex'].append(2)

    if flag == 'train':
        parameter_list['checkpoint_dir'] = parameter_list['checkpoint_dir'] + str(parameter_list['en_width']) + str(parameter_list['kaux_width_real']) + str(parameter_list['kaux_width_complex']) + '_' + '_'.join(map(str, parameter_list['en_units'])) + '_' + '_'.join(map(str, parameter_list['kaux_units_complex']))
        parameter_list['log_dir'] = parameter_list['checkpoint_dir'] + parameter_list['log_dir']
        if not os.path.exists(parameter_list['checkpoint_dir']):
            os.makedirs(parameter_list['log_dir'])

    encoder = net.encoder(parameter_list = parameter_list)
    decoder = net.decoder(parameter_list = parameter_list)
    kaux_real = net.kaux_real(parameter_list)
    kaux_complex = net.kaux_complex(parameter_list)

    return parameter_list, encoder, decoder, kaux_real, kaux_complex 
>>>>>>> First commit for optuna related stuff.
