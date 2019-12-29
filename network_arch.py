import numpy as np
import tensorflow as tf

#Code for encoder layer
class encoder(tf.keras.Model):

    def __init__(self, parameter_list, name= 'ENCODER', return_seq = True):
        super(encoder, self).__init__()
        self.units = parameter_list['en_units']
        self.width = parameter_list['en_width'] - 1
        self.initializer = parameter_list['en_initializer']
        self.output_units = parameter_list['num_evals']
        self.encoder_layer = []

    def build(self, input_shape):
        for i in range(self.width):
            self.encoder_layer.append(tf.keras.layers.Dense(units= self.units, activation= None))
            self.encoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.encoder_layer.append(tf.keras.layers.Dense(units= self.output_units, activation= None))
        self.encoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.layers = tf.keras.Sequential(self.encoder_layer)

    def call(self, inputs):
        return self.layers(inputs)

    def get_config(self):
        configuration = {'Units' : self.units,
                 'Width' : 'LSTM layer = %d' %(self.width),
                 'Activation' : self.activation,
                 'Output' : self.output_units}
        return configuration

#Code for decoder layer
class decoder(tf.keras.Model):

    def __init__(self, parameter_list, name= 'DECODER', return_seq = True):
        super(decoder, self).__init__()
        self.units = parameter_list['de_units']
        self.width = parameter_list['de_width'] - 1
        self.initializer = parameter_list['de_initializer']
        self.output_units = parameter_list['de_output_units']
        self.decoder_layer = []

    def build(self, input_shape):
        for i in range(self.width):
            self.decoder_layer.append(tf.keras.layers.Dense(units= self.units, activation= None))
            self.decoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.decoder_layer.append(tf.keras.layers.Dense(units= self.output_units, activation= None))
        self.decoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.layers = tf.keras.Sequential(self.decoder_layer)

    def call(self, inputs):
        return self.layers(inputs)

    def get_config(self):
        configuration = {'Units' : self.units,
                 'Width' : 'LSTM layer = %d' %(self.width),
                 'Activation' : self.activation,
                 'Output' : self.output_units}
        return configuration

#Code for Koopman Operator Auxilary Network
class koopman_aux_net(tf.keras.Model):

    def __init__(self, parameter_list):
        super(koopman_aux_net, self).__init__()
        self.width = parameter_list['kaux_width']
        self.units = parameter_list['kaux_units']
        self.activation = parameter_list['kp_activation']
        self.koopman_layer_real = []
        self.koopman_layer_complex = []
        self.output_units_real = parameter_list['kaux_output_units_real']
        self.output_units_complex = parameter_list['kaux_output_units_complex']
        self.stateful = parameter_list['stateful']

    def build(self, input_shape):
        #if self.output_units_real:
        for i in range(self.width):
            self.koopman_layer_real.append(tf.keras.layers.LSTM(units= self.units, activation = self.activation, recurrent_activation = 'sigmoid', return_sequences = True, stateful=self.stateful))
        self.koopman_layer_real.append(tf.keras.layers.LSTM(units= self.output_units, activation = self.activation, recurrent_activation = 'sigmoid', return_sequences = True, stateful=self.stateful))

        self.real_layers = tf.keras.Sequential(self.koopman_layer_real)

        #if self.output_units_complex:
        for i in range(self.width):
            self.koopman_layer_complex.append(tf.keras.layers.LSTM(units= self.units, activation = self.activation, recurrent_activation = 'sigmoid', return_sequences = True, stateful=self.stateful))
        self.koopman_layer_complex.append(tf.keras.layers.LSTM(units= self.output_units, activation = self.activation, recurrent_activation = 'sigmoid', return_sequences = True, stateful=self.stateful))

        self.complex_layers = tf.keras.Sequential(self.koopman_layer_real)

    def call(self, inputs):

        #print(f'Calling Koopan_aux_net with input shape {inputs.shape}')
        input_real, input_complex = tf.split(inputs, [self.output_units_real, self.output_units_complex], axis= 2)

        x = self.real_layers(input_real)

        y = self.complex_layers(input_complex)

        return tf.concat([x,y], axis=2)

        def get_config(self):
            configuration = {'Units' : self.units,
                            'Width' : f'Dense layer {self.width}',
                            'Output' : f'Output_real = {self.output_units_real} + Output_complex = {self.output_units_complex}'}
            return configuration

#Code for Koopman Jordarn matrix, input to the layer would be the concatenated omegas for real and complex
#eigenvalues and Koopman embddings (y) from the encoder layer 
class koopman_jordan(tf.keras.Model):

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
            row_1 = tf.squeeze(row_1, 3)

            row_2 = tf.stack([entry_2, entry_1], axis=2)
            row_2 = tf.squeeze(row_2, 3)

            jordan_matrix = tf.stack([row_1, row_2], axis = 3)

            y_jordan_output = tf.stack([y_complex[:, :, index:index+2], y_complex[:, :, index:index+2]], axis = 3)
            y_jordan_output = tf.multiply(jordan_matrix, y_jordan_output)
            y_jordan_output = tf.reduce_sum(y_jordan_output, axis=3)

            y_complex_out.append(y_jordan_output)

        try:
            y_complex_tensor = tf.concat(y_complex_out, axis=2)
        except:
            y_complex_tensor = tf.zeros((inputs.shape[0], inputs.shape[1], 0))

        return tf.concat([y_real_tensor, y_complex_tensor], axis = 2)

class preliminary_net(tf.keras.Model):

    def __init__(self, encoder, decoder, k_net, k_jor, **kwargs):
        super(preliminary_net, self).__init__()
        self.encoder = encoder
        self.koopman_aux_net = k_net
        self.koopman_jordan = k_jor
        self.decoder = decoder

    def call(self, inputs):

        #This part contributes towards the (n+1)th prediction loss from nth
        k_embeddings_cur = self.encoder(inputs)

        k_omegas = self.koopman_aux_net(k_embeddings_cur)
        k_jordan_input = tf.concat([k_omegas, k_embeddings_cur], axis= 2)
        k_jordan_output = self.koopman_jordan(k_jordan_input)

        next_state_space = self.decoder(k_jordan_output) 

        input_reconstruct = self.decoder(k_embeddings_cur)

        return next_state_space, input_reconstruct, k_embeddings_cur, k_jordan_output

#Code for creating the network model
class Koopman_RNN(tf.keras.Model):

    def __init__(self, parameter_list, **kwargs):
        super(Koopman_RNN, self).__init__()
        self.preliminary_net = preliminary_net(parameter_list)

    def call(self, inputs, mth_step = 1):

        next_state_space, input_reconstruct, k_embeddings_cur, k_jordan_output = self.preliminary_net(inputs)

        #This part contributes towards the mth prediction loss
        for_mth_iterations = inputs.shape[1] - mth_step
        next_state_space_mth = tf.TensorArray(tf.float32, size=for_mth_iterations, element_shape = (inputs.shape[0], 1, inputs.shape[2]))
        for i in tf.range(for_mth_iterations):
            inputs_for_mth = inputs[:,i,:]
            inputs_for_mth = tf.expand_dims(inputs_for_mth, axis=1)
            next_state_space_mth_ = tf.zeros_like(inputs_for_mth)

            for j in tf.range(mth_step):
                next_state_space_mth_, _, _, _ = self.preliminary_net(inputs_for_mth)
                inputs_for_mth = next_state_space_mth_

            next_state_space_mth = next_state_space_mth.write(i, next_state_space_mth_)

        try:
            next_state_space_mth = next_state_space_mth.stack()
            next_state_space_mth = tf.squeeze(next_state_space_mth)
            next_state_space_mth = tf.transpose(next_state_space_mth, [1, 0, 2])
        except:
            next_state_space_mth = tf.constant(0, dtype=tf.float32)

        return next_state_space, input_reconstruct, k_embeddings_cur[:,1:,:], k_jordan_output[:,0:-1,:], next_state_space_mth
