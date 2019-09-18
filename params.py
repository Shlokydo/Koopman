parameter_list = {}

#Settings related to dataset creation
parameter_list['key'] = 'nl_pendulum'                   #key for the dataset
parameter_list['num_timesteps'] = 51                    #Number of time steps in one iteration 
parameter_list['Batch_size'] = 256                      #Batch size
parameter_list['Buffer_size'] = 5000                    #Buffer size for shuffle
parameter_list['num_validation_points'] = 1500           #Number of trajectories in the validation set
parameter_list['input_scaling'] = 1                     #Scaling for the input values
parameter_list['delta_t'] = 0.2

#Settings related to network architecture
parameter_list['num_real'] = 0                          #Number of real Koopman eigenvalues
parameter_list['num_complex_pairs'] = 1                 #Number of complex conjugate eigenvalues
parameter_list['num_evals'] = 2                         #Number of output of the network

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
parameter_list['lr_decay_steps'] = 500000                 #No of steps for learning rate decay scheduling
parameter_list['dropout'] = 0.0                         #Dropout for the layers
parameter_list['early_stop_patience'] = 500               #Patience in num of epochs for early stopping
parameter_list['mth_step'] = 30                         #mth step for which prediction needs to be made
parameter_list['mth_cal_patience'] = 1                  #number of epochs after which mth loss is calculated
parameter_list['mth_no_cal_epochs'] = 40                #Number of epochs for which mth loss is not calculated
parameter_list['recon_hp'] = 0.001
parameter_list['global_epoch'] = 0

#Settings related to saving checkpointing and logging
parameter_list['max_checkpoint_keep'] = 4               #Max number of checkpoints to keep
parameter_list['num_epochs_checkpoint'] = 2            #Num of epochs after which to create a checkpoint
parameter_list['log_freq'] = 4                          #Logging frequence for console output
parameter_list['summery_freq'] = 1                      #Logging frequence for summeries
parameter_list['log_dir'] = './summeries'               #Log directory for tensorboard summary
parameter_list['epochs'] = 500                         #Number of epochs
parameter_list['experiments'] = 1                       #Number of experiments to be performed
parameter_list['checkpoint_dir'] = './results'      #Directory to save checkpoint