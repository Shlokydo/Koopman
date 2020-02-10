import tensorflow as tf
import math
import time
import numpy as np
import sys
import os

import network_arch as net 
import Helperfunction as helpfunc 
from plot import plot_figure, plot_diff, animate

def test(pl, model, time_steps = 40, N_traj = 20):

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
    x_diff = helpfunc.difference(x_t_true[:,2:time_steps+1,:], x_t[:,2:,:])
    plot_diff(x_diff[:,:,0], time, True, pl['checkpoint_expdir']+'/media/x_variable.png')
    plot_diff(x_diff[:,:,1], time, True, pl['checkpoint_expdir']+'/media/y_variable.png')
    x_t_true = np.concatenate((x_t_true[:,:time_steps+1,:], x_t), axis=0)
    plot_figure(x_t_true, True, pl['checkpoint_expdir'] + '/media/nl_pendulum.png')
    animate(x_t_true, pl['checkpoint_expdir'] + '/media/video.mp4')
    return None

def train(pl, model, checkpoint, manager, summary_writer, optimizer):
    #Importing dataset into dataframe
    dataframe = helpfunc.import_datset(pl['key'])

    #Converting dataframe to numpy array
    nparray = helpfunc.dataframe_to_nparray(dataframe)

    #Delete the unrequired columns in the nparray
    nparray = helpfunc.nparray_del_vec_along_axis(nparray, 0, ax = 1)

    #Change the datatype to float32
    initial_dataset = helpfunc.change_nparray_datatype(nparray, 'float32')

    #Scale the dataset
    initial_dataset = helpfunc.dataset_scaling(initial_dataset, pl['input_scaling'])

    #Generate split x and y sequence from the dataset
    initial_dataset_x, initial_dataset_y = helpfunc.split_sequences(initial_dataset, pl['num_timesteps'])

    #Shuffling the dataset
    initial_dataset_x, initial_dataset_y = helpfunc.np_array_shuffle(initial_dataset_x, initial_dataset_y)

    #Splitting the dataset into train and validation dataset
    initial_dataset_x_train = initial_dataset_x[:-pl['num_validation_points']]
    initial_dataset_y_train = initial_dataset_y[:-pl['num_validation_points']]

    initial_dataset_x_val = initial_dataset_x[-pl['num_validation_points']:]
    initial_dataset_y_val = initial_dataset_y[-pl['num_validation_points']:]

    #Creating tensorflow dataset for train and val datasets
    tf_dataset_x = helpfunc.create_tfdataset(initial_dataset_x_train)
    tf_dataset_y = helpfunc.create_tfdataset(initial_dataset_y_train)

    tf_dataset_x_val = helpfunc.create_tfdataset(initial_dataset_x_val)
    tf_dataset_y_val = helpfunc.create_tfdataset(initial_dataset_y_val)

    #Batching the dataset
    tf_dataset_batch_x = tf_dataset_x.batch(batch_size= pl['Batch_size'], drop_remainder= True)
    tf_dataset_batch_y = tf_dataset_y.batch(batch_size= pl['Batch_size'], drop_remainder= True)

    tf_dataset_batch_x_val = tf_dataset_x_val.batch(batch_size= pl['Batch_size'], drop_remainder= False)
    tf_dataset_batch_y_val = tf_dataset_y_val.batch(batch_size= pl['Batch_size'], drop_remainder= False)

    #Zipping the dataset
    dataset = tf.data.Dataset.zip((tf_dataset_batch_x, tf_dataset_batch_y))
    val_dataset = tf.data.Dataset.zip((tf_dataset_batch_x_val, tf_dataset_batch_y_val))

    #Shuffling the dataset again (kinda redundant but no problem so why not)
    dataset = dataset.shuffle(pl['Buffer_size'], reshuffle_each_iteration=True)

    #loss function for training 
    loss_func = tf.keras.losses.MeanSquaredError()

    #Metric for loss and validation
    loss_metric_train = tf.keras.metrics.RootMeanSquaredError()
    loss_metric_val = tf.keras.metrics.RootMeanSquaredError()

    #Initialing training variables
    global_step = 0
    global_step_val = 0
    val_min = 0
    val_loss_min = 100

    cal_mth_loss_flag = False
    cal_mth_loss_flag_val = False
    mth_loss_calculation_manipulator = pl['mth_no_cal_epochs']
    s_p = pl['num_timesteps'] - pl['mth_step'] - 1

    #Starting training
    with summary_writer.as_default():
    
        epochs = pl['epochs']
        
        for epoch in range(epochs):
            
            epoch = pl['global_epoch'] + epoch + 1

            start_time = time.time()
            print('\nStart of epoch %d' %(epoch))

            #Manipulating flag for mth prediction loss calculation
            if not((epoch) % mth_loss_calculation_manipulator):
                if not(cal_mth_loss_flag_val):
                    print('\nStarting calculating mth prediction loss\n')
                    # cal_mth_loss_flag = True
                    cal_mth_loss_flag_val = True
                    mth_loss_calculation_manipulator = pl['mth_cal_patience']
                else:
                    # cal_mth_loss_flag = False
                    cal_mth_loss_flag_val = False
                    mth_loss_calculation_manipulator = pl['mth_no_cal_epochs']
                    print('\nStopping calulation of mth prediction loss\n')

            # Iterate over the batches of the dataset.
            for step, (t_current, t_next_actual) in enumerate(dataset):

                global_step += 1

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:

                    t_next_predicted, reconstruction_loss, linearization_loss, t_mth_predictions = model(t_current,
                                                                                                        cal_mth_loss = cal_mth_loss_flag)
                    
                    #Calculating relative loss
                    loss_next_prediction = loss_func(t_next_actual, t_next_predicted) / tf.reduce_mean(tf.square(t_next_actual))
                    if cal_mth_loss_flag:
                        loss_mth_prediction = loss_func(t_mth_predictions, t_next_actual[:, pl['mth_step']:,:]) / tf.reduce_mean(tf.square(t_next_actual[:, pl['mth_step']:,:]))
                        loss = loss_mth_prediction
                    else:
                        loss_mth_prediction = 0
                        loss = loss_next_prediction
                        
                    # loss = loss_next_prediction + (loss_mth_prediction / s_p)
                    loss += sum(model.losses)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                loss_metric_train(t_next_actual, t_next_predicted)

                # Log of train results
                if step % pl['log_freq'] == 0:  
                    print('Training loss (for one batch) at step %s: %s' % (step+1, float(loss)))
                    print('Seen so far: %s samples' % ((global_step) * pl['Batch_size']))
                    
            # Display metrics at the end of each epoch.
            train_acc = loss_metric_train.result()
            print('Training acc over epoch: %s \n' % (float(train_acc),))

            if not(epoch % pl['summery_freq']):
                tf.summary.scalar('RootMSE error', loss_metric_train.result(), step= epoch)
                tf.summary.scalar('Loss_total', loss, step= epoch)
                tf.summary.scalar('Reconstruction_loss', reconstruction_loss, step= epoch)
                tf.summary.scalar('Linearization_loss', linearization_loss, step= epoch)
                if cal_mth_loss_flag:
                    tf.summary.scalar('Mth_prediction_loss', loss_mth_prediction, step= epoch)

            # Reset training metrics at the end of each epoch
            loss_metric_train.reset_states()

            #Code for validation at the end of each epoch
            for step_val, (t_current_val, t_next_actual_val) in enumerate(val_dataset):
                
                global_step_val += 1
                
                t_next_predicted_val, reconstruction_loss_val, linearization_loss_val, t_mth_predictions_val = model(t_current_val, cal_mth_loss = cal_mth_loss_flag_val)
                
                val_loss_next_prediction = loss_func(t_next_predicted_val, t_next_actual_val) / tf.sqrt(tf.reduce_mean(tf.square(t_next_actual_val)))
                if cal_mth_loss_flag_val:
                    val_loss_mth_prediction = loss_func(t_next_actual_val[:, pl['mth_step']:,:], t_mth_predictions_val) / tf.sqrt(tf.reduce_mean(tf.square(t_next_actual_val[:, pl['mth_step']:,:])))


                val_loss = val_loss_next_prediction
                val_loss += sum(model.losses)
                loss_metric_val(t_next_actual_val, t_next_predicted_val)

                if ((step_val) % pl['log_freq']) == 0:
                    print('Validation loss (for one batch) at step %s: %s' % (step_val, float(val_loss)))
                    print('Seen so far: %s samples' % ((step + 1) * pl['Batch_size']))

            val_acc = loss_metric_val.result()
            print('Validation acc over epoch: %s \n' % (float(val_acc),))

            if not(epoch % pl['summery_freq']):
                tf.summary.scalar('RootMSE error_val', loss_metric_val.result(), step= epoch)
                tf.summary.scalar('Loss_total_val', val_loss, step= epoch)
                tf.summary.scalar('Reconstruction_loss_val', reconstruction_loss_val, step= epoch)
                tf.summary.scalar('Linearization_loss_val', linearization_loss_val, step= epoch)
                if cal_mth_loss_flag_val:
                    tf.summary.scalar('Mth_prediction_loss_val', val_loss_mth_prediction, step= epoch)

            # Reset training metrics at the end of each epoch
            loss_metric_val.reset_states()
                    
            if val_loss_min > val_loss:
                val_loss_min = val_loss
                checkpoint.epoch.assign_add(1)
                if int(checkpoint.epoch + 1) % pl['num_epochs_checkpoint'] == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(checkpoint.epoch.numpy(), save_path))
                    print("loss {:1.8f}".format(loss.numpy()))

            if math.isnan(val_acc):
                print('Breaking out as the validation loss is nan')
                break                
            
            if (epoch > 19):
                if not (epoch % pl['early_stop_patience']):
                    if not (val_min):
                        val_min = val_acc
                    else:
                        if val_min > val_acc:
                            val_min = val_acc
                        else:
                            print('Breaking loop as validation accuracy not improving')
                            print("loss {}".format(loss.numpy()))
                            break
            
            epoch_time = (time.time() - start_time)/60

            tf.summary.scalar('Epoch time', epoch_time, step=epoch)

            # print('\nTime for epoch (in minutes): %s \n' %(epoch_time))

    #if not(os.path.exists(pl['model_loc'])):
    #    model_json = model.to_json()
    #    helpfunc.write_to_json(pl['model_loc'], model_json)

    pl['global_epoch'] = epoch 
    return pl['global_epoch']

def traintest(pl, flag):

    #Code for training
    model = get_model(pl)
    #Creating the learning rate scheduler
    learningrate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(pl['learning_rate'],decay_steps = pl['lr_decay_steps'], decay_rate = 0.96, staircase = True)

    #Setting up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=pl['learning_rate'],
                                     beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #Creating summary writer
    summary_writer = tf.summary.create_file_writer(logdir= pl['log_dir'])

    #Creating checkpoint instance
    checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), optimizer = optimizer, model = model)
    save_directory = pl['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory,
                                        max_to_keep= pl['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {} \n".format(manager.latest_checkpoint))

        if flag == 'test':
            print('Starting testing...')
            test(pl, model)
            return pl['global_epoch']

        if flag == 'train':
            print('Starting training of Experiment: {}... \n'.format(pl['Experiment_No']))
            return train(pl, model, checkpoint, manager, summary_writer, optimizer)

    else:
        print("No checkpoint exists.")

        if flag == 'test':
            print('Cannot test as no checkpoint exists. Exiting...')
            return pl['global_epoch']

        if flag == 'train':
            print('Initializing from scratch for Experiment: {}... \n'.format(pl['Experiment_No']))
            return train(pl, model, checkpoint, manager, summary_writer, optimizer)

def get_model(pl):

    print(pl)
    model = net.Koopman_RNN(pl)

    return model
