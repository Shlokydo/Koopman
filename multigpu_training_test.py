import tensorflow as tf
import math
import time
import numpy as np
import sys
import os
import mlflow
import mlflow.tensorflow

mlflow.tensorflow.autolog()

import network_arch as net
import Helperfunction as helpfunc
from plot import plot_figure, plot_diff, animate

import optuna

mirrored_strategy = tf.distribute.MirroredStrategy()

def train(trial, pl, preliminary_net, loop_net, checkpoint, manager, optimizer):

    mlflow.set_experiment(pl['key'])

    #Importing dataset into dataframe
    dataframe = helpfunc.import_datset(pl['dataset'], pl['key'])

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
    initial_dataset_x = initial_dataset_x[:(pl['num_validation_points']+pl['num_training_points'])]
    initial_dataset_y = initial_dataset_y[:(pl['num_validation_points']+pl['num_training_points'])]

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
    tf_dataset_batch_x = tf_dataset_x.batch(batch_size= pl['Batch_size'], drop_remainder= False)
    tf_dataset_batch_y = tf_dataset_y.batch(batch_size= pl['Batch_size'], drop_remainder= False)

    tf_dataset_batch_x_val = tf_dataset_x_val.batch(batch_size= pl['Batch_size_val'], drop_remainder= False)
    tf_dataset_batch_y_val = tf_dataset_y_val.batch(batch_size= pl['Batch_size_val'], drop_remainder= False)

    #Zipping the dataset
    dataset = tf.data.Dataset.zip((tf_dataset_batch_x, tf_dataset_batch_y))
    val_dataset = tf.data.Dataset.zip((tf_dataset_batch_x_val, tf_dataset_batch_y_val))

    #Shuffling the dataset again (kinda redundant but no problem so why not)
    dataset = dataset.shuffle(pl['Buffer_size'], reshuffle_each_iteration=True)

    #Distributing the dataset
    dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)

    decaying_weights = np.asarray([math.pow(pl['l_decay_param'], j) for j in range(pl['num_timesteps'] - 1)])
    decaying_weights_sum = np.sum(decaying_weights)
    decaying_weights = decaying_weights/decaying_weights_sum
    decaying_weights = tf.convert_to_tensor(decaying_weights, dtype = 'float32')

    try:
        rname = str(trial.study.study_name) + '_' + str(trial.number)
    except:
        rname = 'best'

    with mlflow.start_run(run_name = rname):

        mlflow.log_params(pl)

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

                return per_example_loss / (pl['Batch_size'] * (pl['num_timesteps'] - 1))

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
                    linearization_loss = compute_loss(t_embedding, t_jordan, weighted = 0) / pl['num_timesteps'] 
                    #linearization_loss = compute_loss(t_embedding, t_jordan, weighted = pl['weighted'])  

                gradients = tape.gradient([loss_next_prediction, linearization_loss, reconstruction_loss], preliminary_net.trainable_variables)
                optimizer.apply_gradients(zip(gradients, preliminary_net.trainable_weights))           

                if mth_flag:
                    with tf.GradientTape() as tape:     
                        t_mth_predictions = loop_net(t_current)
                        loss_mth = loss_func(t_mth_predictions, t_next_actual[:, pl['mth_step']:,:]) / (pl['Batch_size'] * (pl['num_timesteps'] - pl['mth_step'] - 1)) * pl['mth_mellow'] 
                    
                    loop_net.layers[0].trainable=False
                    loop_net.layers[3].trainable=False
                    gradients = tape.gradient([loss_mth], loop_net.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, loop_net.trainable_weights))           
                    loop_net.layers[0].trainable=True
                    loop_net.layers[3].trainable=True
                else:
                    loss_mth = tf.constant(0, dtype=tf.float32)

                metric_device = compute_metric(t_next_actual, t_next_predicted)
                metric.reset_states()

                return loss_next_prediction, metric_device, reconstruction_loss, linearization_loss, loss_mth

            def val_step(inputs):

                (t_current, t_next_actual), mth_flag = inputs

                t_next_predicted, t_reconstruction, t_embedding, t_jordan = preliminary_net(t_current)

                #Calculating relative loss
                loss_next_prediction = compute_loss(t_next_actual, t_next_predicted, 0) * 0.001
                reconstruction_loss = compute_loss(t_current, t_reconstruction, 0) * 0.001
                linearization_loss = compute_loss(t_embedding, t_jordan, 0) / pl['num_timesteps'] 

                if mth_flag:
                    t_mth_predictions = loop_net(t_current)
                    loss_mth = loss_func(t_mth_predictions, t_next_actual[:, pl['mth_step']:,:]) / (pl['Batch_size'] * (pl['num_timesteps'] - pl['mth_step'] - 1))
                else:
                    loss_mth = tf.constant(0, dtype=tf.float32)

                metric_device = compute_metric(t_next_actual, t_next_predicted)
                metric.reset_states()

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
            global_epoch = pl['global_epoch']
            global_step = 0
            val_min = 0
            val_loss_min = pl['val_min']
            timer_tot = time.time()

            cal_mth_loss_flag = False
            mth_loss_cal_mani = pl['mth_no_cal_epochs']

            epochs = pl['epochs']

            for epoch in range(epochs):

                global_epoch += 1

                start_time = time.time()
                print('\nStart of epoch {}'.format(global_epoch))

                if (epoch > pl['only_RNN']):
                    if not((global_epoch) % mth_loss_cal_mani):
                        if not(cal_mth_loss_flag):
                            print("\n Starting calculation of mth prediction loss \n")
                            cal_mth_loss_flag = True
                            mth_loss_cal_mani = pl['mth_cal_patience']
                        else:
                            cal_mth_loss_flag = False
                            mth_loss_cal_mani = pl['mth_no_cal_epochs']
                            print("Stopping mth calculation loss")

                # Iterate over the batches of the dataset.
                for step, inp in enumerate(dataset):

                    global_step += 1

                    inputs = (inp, cal_mth_loss_flag) 
                    loss, t_metric, t_reconst, t_lin, t_mth = distributed_train(inputs)

                    if not (step % pl['log_freq']):
                        print('Training loss (for one batch) at step {}: {}'.format(step+1, float(loss)))
                        print('Seen so far: {} samples'.format(global_step * pl['Batch_size']))

                print('Training acc over epoch: {} \n'.format(float(t_metric)))

                for v_step, v_inp in enumerate(val_dataset):

                    v_inputs = (v_inp, cal_mth_loss_flag)
                    v_loss, v_metric, v_reconst, v_lin, v_mth = distributed_val(v_inputs)

                    if not (v_step % pl['log_freq']):
                        print('Validation loss (for one batch) at step {} : {}'.format(v_step + 1, float(v_loss)))

                print('Validation acc over epoch: {} \n'.format(float(v_metric)))

                trial.report(v_loss, epoch)
                trial.report(v_reconst, epoch)
                trial.report(v_lin, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if val_loss_min > (v_loss + v_reconst + v_lin):
                    val_loss_min = (v_loss + v_reconst + v_lin)

                    if (global_epoch > pl['only_RNN']):
                        # Report intermediate objective value.
                        checkpoint.epoch.assign(global_epoch)
                        save_path = manager.save()
                        print("Saved checkpoint for epoch {}: {}".format(checkpoint.epoch.numpy(), save_path))
                        print("loss {}".format(loss.numpy()))

                if not (global_epoch % pl['summery_freq']):
                    mlflow.log_metric('V_RMSE', v_metric.numpy(), step= global_epoch)
                    mlflow.log_metric('V_MSE_RNN', v_loss.numpy(), step= global_epoch)
                    mlflow.log_metric('V_Reconstruct', v_reconst.numpy(), step= global_epoch)
                    mlflow.log_metric('V_Linearization', v_lin.numpy(), step= global_epoch)

                    mlflow.log_metric('T_RMSE', t_metric.numpy(), step= global_epoch)
                    mlflow.log_metric('T_MSE_RNN', loss.numpy(), step= global_epoch)
                    mlflow.log_metric('T_Reconstruct', t_reconst.numpy(), step= global_epoch)
                    mlflow.log_metric('T_Linearization', t_lin.numpy(), step= global_epoch)

                    if cal_mth_loss_flag:
                        mlflow.log_metric('V_Mth', v_mth.numpy(), step= global_epoch)
                        mlflow.log_metric('T_Mth', t_mth.numpy(), step= global_epoch)

                    mlflow.log_metric('LearnRate', optimizer._decayed_lr(var_dtype=tf.float32).numpy())
                    mlflow.log_metric('Total_loss', val_loss_min.numpy(), step = global_epoch)

                if math.isnan(v_metric):
                    print('Breaking out as the validation loss is nan')
                    break

                print('Time for epoch (in seconds): %s' %((time.time() - start_time)))

        print('\n Total Epoch time (in minutes): {}'.format((time.time()-timer_tot)/60))
        pl['global_epoch'] = global_epoch
        pl['val_min'] = val_loss_min.numpy()
        print('\nNumber of iterations for optimizer: {}'.format(optimizer.iterations.numpy()))

        helpfunc.write_pickle(pl, pl['pickle_name'])
        mlflow.log_artifact(pl['pickle_name'])
        mlflow.log_artifact('./test_optuna.sh')
        mlflow.log_params(pl)
        mlflow.set_tags(trial.params)
        mlflow.end_run()
    
    return val_loss_min

def traintest(trial, pl):

    print('\nGPU Available: {}\n'.format(tf.test.is_gpu_available()))

    #Get the Model
    with mirrored_strategy.scope():

        encoder = net.encoder(pl = pl)
        decoder = net.decoder(pl = pl)
        kaux_real = net.kaux_real(pl)
        kaux_complex = net.kaux_complex(pl)
        koopman_aux_net = net.koopman_aux_net(kaux_real, kaux_complex, pl = pl)
        koopman_jordan = net.koopman_jordan(pl = pl)

        preliminary_net = net.preliminary_net(pl, encoder, decoder, koopman_aux_net, koopman_jordan)
        loop_net = net.loop_net(pl, encoder, decoder, koopman_aux_net, koopman_jordan)

        #Defining Model compiling parameters
        learningrate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(pl['learning_rate'], decay_steps = pl['lr_decay_steps'], decay_rate = pl['lr_decay_rate'], staircase = False)
        #learning_rate = learningrate_schedule
        learning_rate = pl['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #Defining the checkpoint instance
        checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), encoder = encoder, decoder = decoder, kaux_real = kaux_real, kaux_complex = kaux_complex, koopman_jordan = koopman_jordan, optimizer = optimizer)

    #Creating checkpoint instance
    save_directory = pl['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, max_to_keep= pl['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        print('Starting training of Experiment... \n')
        to_return = train(trial, pl, preliminary_net, loop_net, checkpoint, manager, optimizer)
        print('\nMinimum val_loss calculated: {}\n'.format(to_return))
        return to_return

    else:
        print("No checkpoint exists.")

        print('Initializing from scratch for Experiment... \n')
        to_return = train(trial, pl, preliminary_net, loop_net, checkpoint, manager, optimizer)
        print('\nMinimum val_loss calculated: {}\n'.format(to_return))
        return to_return
