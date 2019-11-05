import tensorflow as tf
import math
import time
import numpy as np
import sys
import os

import network_arch as net 
import Helperfunction as helpfunc 
from plot import plot_figure, plot_diff, animate

mirrored_strategy = tf.distribute.MirroredStrategy()

def test(parameter_list, model, time_steps = 5, N_traj = 3):

    time, x_t_true = helpfunc.nl_pendulum(N= N_traj)
    prediction_list_global = []

    for j in range(x_t_true.shape[0]):
        prediction_list_local = []
        input_value = helpfunc.input_generator(x_t_true[j,0,:])
        prediction_list_local.append(input_value[0,0,:])
        for i in range(time_steps):
            prediction, _, _, _ = model(input_value, cal_mth_loss = False)
            input_value = prediction
            prediction_list_local.append(prediction.numpy()[0,0,:])
        x_t_local = np.asarray(prediction_list_local)
        prediction_list_global.append(prediction_list_local)

    x_t = np.asarray(prediction_list_global)
    x_diff = helpfunc.difference(x_t_true[:,:time_steps+1,:], x_t)
    plot_diff(x_diff[:,:,0], time, True, './x_variable.png')
    plot_diff(x_diff[:,:,1], time, True, './y_variable.png')
    x_t_true = np.concatenate((x_t_true[:,:time_steps+1,:], x_t), axis=0)
    animate(x_t_true)
    plot_figure(x_t_true, True)
    return None

def train(parameter_list, model, checkpoint, manager, summary_writer, optimizer):
    #Importing dataset into dataframe
    dataframe = helpfunc.import_datset(parameter_list['key'])

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
    tf_dataset_batch_x = tf_dataset_x.batch(batch_size= parameter_list['Batch_size'], drop_remainder= True)
    tf_dataset_batch_y = tf_dataset_y.batch(batch_size= parameter_list['Batch_size'], drop_remainder= True)

    tf_dataset_batch_x_val = tf_dataset_x_val.batch(batch_size= parameter_list['Batch_size'], drop_remainder= False)
    tf_dataset_batch_y_val = tf_dataset_y_val.batch(batch_size= parameter_list['Batch_size'], drop_remainder= False)

    #Zipping the dataset
    dataset = tf.data.Dataset.zip((tf_dataset_batch_x, tf_dataset_batch_y))
    val_dataset = tf.data.Dataset.zip((tf_dataset_batch_x_val, tf_dataset_batch_y_val))

    #Shuffling the dataset again (kinda redundant but no problem so why not)
    dataset = dataset.shuffle(parameter_list['Buffer_size'], reshuffle_each_iteration=True)

    #Distributing the dataset
    dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)

    with mirrored_strategy.scope():

        #loss function for training 
        loss_func = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM, name='LossMSE')

        #Metric
        metric = tf.keras.metrics.RootMeanSquaredError(name='MetricRMSE')

        def compute_loss(labels, predictions):
            per_example_loss = loss_func(labels, predictions)
            return per_example_loss * (1.0 / (parameter_list['Batch_size'] * (parameter_list['num_timesteps'] - 1)))

        def compute_metric(labels, predictions):
            per_example_metric = metric(labels, predictions)
            return per_example_metric #* (1.0 / (parameter_list['Batch_size'] * parameter_list['timesteps']))

        def train_step(inputs):
            with tf.GradientTape() as tape:

                (t_current, t_next_actual), cal_mth_loss_flag = inputs
                t_next_predicted, reconstruction_loss, linearization_loss, t_mth_predictions = model(t_current, cal_mth_loss = cal_mth_loss_flag)

                #Calculating relative loss
                loss_next_prediction = compute_loss(t_next_actual, t_next_predicted) #/ tf.reduce_mean(tf.square(t_next_actual))
                if cal_mth_loss_flag:
                    loss_mth_prediction = loss_func(t_mth_predictions, t_next_actual[:, parameter_list['mth_step']:,:]) / (1.0 / (parameter_list['Batch_size'] * (parameter_list['num_timesteps'] - parameter_list['mth_step'])))
                    loss = loss_mth_prediction
                else:
                    loss_mth_prediction = 0
                    loss = loss_next_prediction
                    
                # loss = loss_next_prediction + (loss_mth_prediction / s_p)
                loss += sum(model.losses)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            metric_device = compute_metric(t_next_actual, t_next_predicted)

            return loss, metric_device, reconstruction_loss, linearization_loss, loss_mth_prediction

        def val_step(inputs):
            (t_current_val, t_next_actual_val), cal_mth_loss_flag_val = inputs

            t_next_predicted_val, reconstruction_loss_val, linearization_loss_val, t_mth_predictions_val = model(t_current_val, cal_mth_loss = cal_mth_loss_flag_val)

            val_loss_next_prediction = compute_loss(t_next_predicted_val, t_next_actual_val)
            if cal_mth_loss_flag_val:
                val_loss_mth_prediction = loss_func(t_mth_prediction_val, t_next_actual_val[:, parameter_list['mth_step']:,:]) / (1.0 / (parameter_list['Batch_size'] * (parameter_list['num_timesteps'] - parameter_list['mth_step'])))
            else:
                val_loss_mth_prediction = 0

            val_loss = val_loss_next_prediction
            val_loss += sum(model.losses)

            val_metric_device = compute_metric(t_next_predicted_val, t_next_actual_val)

            return val_loss, val_metric_device, reconstruction_loss_val, linearization_loss_val, val_loss_mth_prediction

        @tf.function
        def distributed_train(inputs):

            pr_losses, pr_metric, pr_reconst, pr_lin, pr_mth = mirrored_strategy.experimental_run_v2(train_step, args=(inputs,))
            
            losses = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_losses, axis=None)
            metric = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metric, axis=None)
            reconst = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_reconst, axis=None)
            lin = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_lin, axis=None)
            mth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_mth, axis=None)

            return losses, metric, reconst, lin, mth

        @tf.function
        def distributed_val(inputs):
            pr_losses, pr_metric, pr_reconst, pr_lin, pr_mth = mirrored_strategy.experimental_run_v2(val_step, args=(inputs,))

            losses = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_losses, axis=None)
            metric = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metric, axis=None)
            reconst = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_reconst, axis=None)
            lin = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_lin, axis=None)
            mth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pr_mth, axis=None)

            return losses, metric, reconst, lin, mth

        #Initialing training variables
        global_epoch = parameter_list['global_epoch']
        global_step = 0
        val_min = 0
        val_loss_min = parameter_list['val_min']

        cal_mth_loss_flag = 0
        cal_mth_loss_flag_val = 0
        mth_loss_calculation_manipulator = parameter_list['mth_no_cal_epochs']
        s_p = parameter_list['num_timesteps'] - parameter_list['mth_step'] - 1

        with summary_writer.as_default():

            epochs = parameter_list['epochs']

            for epoch in range(epochs):

                global_epoch += 1

                start_time = time.time()
                print('\nStart of epoch {}'.format(global_epoch))

                #Manipulating flag for mth prediction loss calculation
                if not((global_epoch) % mth_loss_calculation_manipulator):
                    if not(cal_mth_loss_flag_val):
                        print('\nStar ting calculating mth prediction loss\n')
                        # cal_mth_loss_flag = 1
                        cal_mth_loss_flag_val = 1
                        mth_loss_calculation_manipulator = parameter_list['mth_cal_patience']
                    else:
                        # cal_mth_loss_flag = 0
                        cal_mth_loss_flag_val = 0
                        mth_loss_calculation_manipulator = parameter_list['mth_no_cal_epochs']
                        print('\nStopping calulation of mth prediction loss\n')

                # Iterate over the batches of the dataset.
                for step, inp in enumerate(dataset):
                    
                    inputs = (inp, cal_mth_loss_flag)

                    global_step += 1 

                    # Open a GradientTape to record the operations run
                    # during the forward pass, which enables autodifferentiation.
                    loss, t_metric, t_reconst, t_lin, t_mth = distributed_train(inputs)

                    if not (step % parameter_list['log_freq']):
                        print('Training loss (for one batch) at step {}: {}'.format(step+1, float(loss)))
                        print('Seen so  far: {} samples'.format(global_step * parameter_list['Batch_size']))

                print('Training acc over epoch: {} \n'.format(float(t_metric)))

                if not (global_epoch % parameter_list['summery_freq']):
                    tf.summary.scalar('RootMSE error', t_metric, step= global_epoch)
                    tf.summary.scalar('Loss_total', loss, step= global_epoch)
                    tf.summary.scalar('Reconstruction_loss', t_reconst, step= global_epoch)
                    tf.summary.scalar('Linearization_loss', t_lin, step= global_epoch)
                    if cal_mth_loss_flag:
                        tf.summary.scalar('Mth_prediction_loss', t_mth, step= global_epoch)

                for v_step, v_inp in enumerate(val_dataset):

                    v_inputs = (v_inp, cal_mth_loss_flag_val)

                    v_loss, v_metric, v_reconst, v_lin, v_mth = distributed_val(v_inputs) 

                    if not (v_step % parameter_list['log_freq']):   
                        print('Validation loss (for one batch) at step {} : {}'.format(v_step + 1, float(v_loss)))
                
                print('Validation acc over epoch: {} \n'.format(float(v_metric)))

                if not (global_epoch % parameter_list['summery_freq']):
                    tf.summary.scalar('RootMSE error', v_metric, step= global_epoch)
                    tf.summary.scalar('Loss_total', v_loss, step= global_epoch)
                    tf.summary.scalar('Reconstruction_loss', v_reconst, step= global_epoch)
                    tf.summary.scalar('Linearization_loss', v_lin, step= global_epoch)
                    if cal_mth_loss_flag_val:
                        tf.summary.scalar('Mth_prediction_loss', v_mth, step= global_epoch)

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

                if (global_epoch > 19):
                    if not (epoch % parameter_list['early_stop_pati ence']):
                        if  not (val_min):
                            val_min = v_metric
                        else:
                            if val_min > val_acc: 
                                val_min = val_acc
                            else:
                                print('Breaking loop  as validation accuracy not improving')
                                print("loss {}".format(loss.numpy()))
                                break

                print('Time for epoch (in seconds): %s' %((time.time() - start_time)))

    #if not(os.path.exists(parameter_list['model_loc'])):
    #    model_json = model.to_json()
    #    helpfunc.write_to_json(parameter_list['model_loc'], model_json)

    parameter_list['global_epoch'] = global_epoch 
    parameter_list['val_min'] = val_loss_min
    return parameter_list

def traintest(parameter_list, flag):

    print('\nGPU Available: {}\n'.format(tf.test.is_gpu_available()))

    #Get the Model
    with mirrored_strategy.scope():
    #    if os.path.exists(parameter_list['model_loc']):
    #        print('\nLoading saved model...\n')
    #        j_string = helpfunc.read_json(parameter_list['model_loc'])
    #        model = tf.keras.models.model_from_json(j_string)
    #    else:
        model = net.Koopman_RNN(parameter_list = parameter_list)

        #Defining Model compiling parameters
        learningrate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(parameter_list['learning_rate'], decay_steps = parameter_list['lr_decay_steps'], decay_rate = parameter_list['lr_decay_rate'], staircase = True)
        learning_rate = learningrate_schedule 
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #Defining the checkpoint instance
        checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), model = model, optimizer = optimizer)

    #Creating summary writer
    summary_writer = tf.summary.create_file_writer(logdir= parameter_list['log_dir'])

    #Creating checkpoint instance
    save_directory = parameter_list['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, 
                                        max_to_keep= parameter_list['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        if flag == 'test':
            print('Starting testing...')
            test(parameter_list, model)
            return parameter_list
            
        if flag == 'train':
            print('Starting training of Experiment: {}... \n'.format(parameter_list['Experiment_No']))
            return train(parameter_list, model, checkpoint, manager, summary_writer, optimizer)

    else:
        print("No checkpoint exists.")
        
        if flag == 'test':
             print('Cannot test as no checkpoint exists. Exiting...')
             sys.exit()
             return parameter_list
        
        if flag == 'train':
            print('Initializing from scratch for Experiment: {}... \n'.format(parameter_list['Experiment_No']))
            return train(parameter_list, model, checkpoint, manager, summary_writer, optimizer)

    print(learning_rate)
