import numpy as np 
import tensorflow as tf 

#Code for encoder layer

class encoder(tf.keras.layers.Layer):
    
    def __init__(self, parameter_list, name= 'ENCODER', return_seq = True):
        super(encoder, self).__init__()
        self.units = parameter_list['en_units']
        self.width = parameter_list['en_width'] - 1
        self.initializer = parameter_list['en_initializer']
        self.activation = parameter_list['en_activation']
        self.output_units = parameter_list['num_evals']
        self.encoder_layer = []
        
    def build(self, input_shape):
        for i in range(self.width):
            self.encoder_layer.append(tf.keras.layers.LSTM(units= self.units, activation= self.activation,
                                                      recurrent_activation = 'sigmoid', return_sequences = True))
                                                      #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                                      #dropout = 0))
            
        self.encoder_layer.append(tf.keras.layers.LSTM(units= self.output_units, activation= self.activation,
                                                      recurrent_activation = 'sigmoid', return_sequences = True))
                                                      #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                                      #dropout = 0))
    def call(self, inputs):
        for i in range(len(self.encoder_layer)):
            if i == 0:
                x = self.encoder_layer[i](inputs)
            else:
                x = self.encoder_layer[i](x)
        return x
    
    def get_config(self):
        configuration = {'Units' : self.units,
                 'Width' : 'LSTM layer = %d' %(self.width),
                 'Activation' : self.activation,
                 'Output' : self.output_units}
        return configuration

#Code for decoder layer

class decoder(tf.keras.layers.Layer):
    
    def __init__(self, parameter_list, name= 'DECODER', return_seq = True):
        super(decoder, self).__init__()
        self.units = parameter_list['de_units']
        self.width = parameter_list['de_width'] - 1
        self.initializer = parameter_list['de_initializer']
        self.activation = parameter_list['de_activation']
        self.output_units = parameter_list['de_output_units']
        self.decoder_layer = []
        
    def build(self, input_shape):
        for i in range(self.width):
            # self.decoder_layer.append(tf.keras.layers.LSTM(units= self.units, activation= self.activation,
                                                    #   recurrent_activation = 'sigmoid', return_sequences = True))
                                                      #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                                      #dropout = 0))
            self.decoder_layer.append(tf.keras.layers.Dense(units= self.units,
                                                            activation= None))
            # self.decoder_layer.append(tf.keras.layers.PReLU(alpha_initializer='zeros'))
            self.decoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))
            
        # self.decoder_layer.append(tf.keras.layers.LSTM(units= self.output_units, activation= self.activation,
                                                    #   recurrent_activation = 'sigmoid', return_sequences = True))
                                                      #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                                      #dropout = 0))

        self.decoder_layer.append(tf.keras.layers.Dense(units= self.output_units,
                                                            activation= None))
        # self.decoder_layer.append(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        self.decoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))
            
    def call(self, inputs):
        for i in range(len(self.decoder_layer)):
            if i == 0:
                x = self.decoder_layer[i](inputs)
            else:
                x = self.decoder_layer[i](x)
        return x
    
    def get_config(self):
        configuration = {'Units' : self.units,
                 'Width' : 'LSTM layer = %d' %(self.width),
                 'Activation' : self.activation,
                 'Output' : self.output_units}
        return configuration

#Code for Koopman Operator Auxilary Network
class koopman_aux_net(tf.keras.layers.Layer):
    
    def __init__(self, parameter_list):
        super(koopman_aux_net, self).__init__()
        self.width = parameter_list['kaux_width']
        self.units = parameter_list['kaux_units']
        self.koopman_layer_real = []
        self.koopman_layer_complex = []
        self.output_units_real = parameter_list['kaux_output_units_real']
        self.output_units_complex = parameter_list['kaux_output_units_complex']
        
    def build(self, input_shape):
        if self.output_units_real:
            for i in range(self.width):
                self.koopman_layer_real.append(tf.keras.layers.Dense(units= self.units,
                                                                activation= None))
                                                                #kernel_regularizer = tf.keras.regularizers.l2(0.01)))
                self.koopman_layer_real.append(tf.keras.layers.LeakyReLU(alpha= 0.3))
                
            self.koopman_layer_real.append(tf.keras.layers.Dense(units= self.output_units_real, 
                                                        activation= None))
                                                       #kernel_regularizer = tf.keras.regularizers.l2(0.01)))
            self.koopman_layer_real.append(tf.keras.layers.LeakyReLU(alpha= 0.3))
            
        if self.output_units_complex:
            for i in range(self.width):
                self.koopman_layer_complex.append(tf.keras.layers.Dense(units= self.units,
                                                                activation= None))
                                                                #kernel_regularizer = tf.keras.regularizers.l2(0.01)))
                self.koopman_layer_complex.append(tf.keras.layers.LeakyReLU(alpha= 0.3))
                
            self.koopman_layer_complex.append(tf.keras.layers.Dense(units= self.output_units_complex, 
                                                        activation= None))
                                                       #kernel_regularizer = tf.keras.regularizers.l2(0.01)))
            self.koopman_layer_complex.append(tf.keras.layers.LeakyReLU(alpha= 0.3))
        
    def call(self, inputs):
        
        x = 0
        y = 0
        # print(f'Calling Koopan_aux_net with input shape {inputs.shape}')
        input_real, input_complex = tf.split(inputs, [self.output_units_real, self.output_units_complex], axis= 2)
        
        for i in range(len(self.koopman_layer_real)):
            if i == 0:
                x = self.koopman_layer_real[i](input_real)
            else:
                x = self.koopman_layer_real[i](x)

        for i in range(len(self.koopman_layer_complex)):
            if i == 0:
                y = self.koopman_layer_complex[i](input_complex)
            else:
                y = self.koopman_layer_complex[i](y)
    
        if x:
            if y:
                return tf.concat([x, y], axis=2)
            else:
                return x
        else:
            return y
        
        def get_config(self):
            configuration = {'Units' : self.units,
                            'Width' : f'Dense layer {self.width}',
                            'Output' : f'Output_real = {self.output_units_real} + Output_complex = {self.output_units_complex}'}
            return configuration

#Code for Koopman Jordarn matrix, input to the layer would be the concatenated omegas for real and complex 
#eigenvalues and Koopman embddings (y) from the encoder layer

class koopman_jordan(tf.keras.layers.Layer):
    
    def __init__(self, parameter_list):
        super(koopman_jordan, self).__init__()
        self.omegas_real = parameter_list['num_real']
        self.omegas_complex = parameter_list['num_complex_pairs'] * 2
        self.delta_t = parameter_list['delta_t']
        
    def call(self, inputs):
        
        # print(f'Calling koopman_jordan with input shape {inputs.shape}')
        omegas_real_vec, omegas_complex_vec, y = tf.split(inputs, [self.omegas_real, self.omegas_complex, self.omegas_real + self.omegas_complex], axis= 2)
        y_real, y_complex = tf.split(y, [self.omegas_real, self.omegas_complex], axis= 2)
        
        scale_real = tf.exp(omegas_real_vec * self.delta_t)
        y_real_tensor = tf.multiply(scale_real, y_real)
        
        y_complex_out = []
        
        for i in range(int(self.omegas_complex/2)):
            index = i * 2
            
            scale_complex = tf.exp(omegas_complex_vec[:, :, index:index + 1] * self.delta_t)
            entry_1 = tf.multiply(scale_complex, tf.cos(omegas_complex_vec[:, :, index + 1: index + 2] * self.delta_t))
            entry_2 = tf.multiply(scale_complex, tf.sin(omegas_complex_vec[:, :, index + 1: index + 2] * self.delta_t))
            row_1 = tf.stack([entry_1, -entry_2], axis=2)
            row_1 = tf.reshape(row_1, [row_1.shape[0], row_1.shape[1], row_1.shape[2]])
            
            row_2 = tf.stack([entry_2, entry_1], axis=2)
            row_2 = tf.reshape(row_2, [row_2.shape[0], row_2.shape[1], row_2.shape[2]])
            
            jordan_matrix = tf.stack([row_1, row_2], axis = 3)
            
            y_jordan_output = tf.stack([y[:, :, index:index+2], y[:, :, index:index+2]], axis = 3)
            y_jordan_output = tf.multiply(jordan_matrix, y_jordan_output)
            y_jordan_output = tf.reduce_sum(y_jordan_output, axis=3)
            
            y_complex_out.append(y_jordan_output)
                        
        for i in range(len(y_complex_out)):
            if i == 0:
                y_complex_tensor = y_complex_out[i]
            else:
                y_complex_tensor = tf.concat([y_complex_tensor, y_complex_out[i]], axis= 2)
                
        return tf.concat([y_real_tensor, y_complex_tensor], axis = 2)

#Code for creating the network model
class Koopman_RNN(tf.keras.Model):

    def __init__(self, parameter_list, **kwargs):
        super(Koopman_RNN, self).__init__()
        self.encoder = encoder(parameter_list)
        self.koopman_aux_net = koopman_aux_net(parameter_list)
        self.koopman_jordan = koopman_jordan(parameter_list)
        self.decoder = decoder(parameter_list)
        self.mth_step = parameter_list['mth_step']
        self.recon_hp = parameter_list['recon_hp']
        self.timesteps = parameter_list['num_timesteps'] - 1
        self.batchsize = parameter_list['Batch_size']

    def call(self, inputs, cal_mth_loss = False):

        #This part contributes towards the (n+1)th prediction loss from nth
        k_embeddings_cur = self.encoder(inputs)

        k_omegas = self.koopman_aux_net(k_embeddings_cur)
        k_jordan_input = tf.concat([k_omegas, k_embeddings_cur], axis= 2)
        k_jordan_output = self.koopman_jordan(k_jordan_input)

        next_state_space = self.decoder(k_jordan_output) + inputs

        #This part contributes towards the reconstruction loss
        input_reconstruct = self.decoder(k_embeddings_cur)
        reconstruct_loss = tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(input_reconstruct, inputs)),1)) / tf.reduce_mean(tf.reduce_mean(tf.square(inputs),1))
        reconstruct_loss = reconstruct_loss * (1.0 / (self.timesteps * self.batchsize))

        #This part contributes towards the linearization loss
        linearization_loss = tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(k_embeddings_cur[:,1:,:], k_jordan_output[:,0:-1,:])), 1))
        linearization_loss = linearization_loss * (1.0 / (self.timesteps * self.batchsize))
        
        reconst_linear_loss = self.recon_hp * reconstruct_loss + linearization_loss
        self.add_loss(reconst_linear_loss)
        next_state_space_mth_list = []
        
        #This part contributes towards the mth prediction loss
        if cal_mth_loss:
            for_mth_iterations = inputs.shape[1] - self.mth_step

            for j in range(for_mth_iterations):
                inputs_for_mth = inputs[:,j,:] 
                inputs_for_mth = tf.expand_dims(inputs_for_mth, axis=1)

                for i in range(self.mth_step):
                    k_embeddings_cur_mth = self.encoder(inputs_for_mth) 

                    k_omegas_mth = self.koopman_aux_net(k_embeddings_cur_mth)
                    k_jordan_input_mth = tf.concat([k_omegas_mth, k_embeddings_cur_mth], axis= 2)
                    k_jordan_output_mth = self.koopman_jordan(k_jordan_input_mth)
                    
                    next_state_space_mth = self.decoder(k_jordan_output_mth)

                    inputs_for_mth = next_state_space_mth

                next_state_space_mth_list.append(next_state_space_mth)
            next_state_space_mth_list = tf.stack(next_state_space_mth_list)
            next_state_space_mth_list = tf.squeeze(next_state_space_mth_list)
            next_state_space_mth_list = tf.transpose(next_state_space_mth_list, perm=[1,0,2])

        reconst_linear_loss = self.recon_hp * reconstruct_loss + linearization_loss

        self.add_loss(reconst_linear_loss)

        return next_state_space, reconstruct_loss, linearization_loss, next_state_space_mth_list
