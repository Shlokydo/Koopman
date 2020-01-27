import numpy as np
import tensorflow as tf

#Code for encoder layer
class encoder(tf.keras.Model):

    def __init__(self, parameter_list, name= 'ENCODER'):
        super(encoder, self).__init__()
        self.units = parameter_list['en_units']
        self.width = parameter_list['en_width']
        self.initializer = parameter_list['en_initializer']
        self.output_units = parameter_list['num_evals']
        self.encoder_layer = []

    def build(self, input_shape):
        for i in range(self.width):
            self.encoder_layer.append(tf.keras.layers.Dense(units= self.units[i], activation= None))
            self.encoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.encoder_layer.append(tf.keras.layers.Dense(units= self.output_units, activation= None))
        self.encoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.layer_net = tf.keras.Sequential(self.encoder_layer)

    def call(self, inputs):
        #print(f'Calling Encoder with input shape {inputs.shape}')
        ret = self.layer_net(inputs)
        #print('Shape of the returning tensor: {}'.format(ret.shape))
        return ret

    def get_config(self):
        configuration = {'Units' : self.units,
                 'Width' : 'LSTM layer = %d' %(self.width),
                 'Activation' : self.activation,
                 'Output' : self.output_units}
        return configuration

#Code for decoder layer
class decoder(tf.keras.Model):

    def __init__(self, parameter_list, name= 'DECODER'):
        super(decoder, self).__init__()
        self.units = parameter_list['de_units']
        self.width = parameter_list['de_width']
        self.initializer = parameter_list['de_initializer']
        self.output_units = parameter_list['de_output_units']
        self.decoder_layer = []

    def build(self, input_shape):
        for i in range(self.width):
            self.decoder_layer.append(tf.keras.layers.Dense(units= self.units[i], activation= None))
            self.decoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.decoder_layer.append(tf.keras.layers.Dense(units= self.output_units, activation= None))
        self.decoder_layer.append(tf.keras.layers.LeakyReLU(alpha= 0.3))

        self.layer_net = tf.keras.Sequential(self.decoder_layer)

    def call(self, inputs):
        #print(f'Calling Decoder with input shape {inputs.shape}')
        ret = self.layer_net(inputs)
        #print('Shape of the returning tensor: {}'.format(ret.shape))
        return ret

    def get_config(self):
        configuration = {'Units' : self.units,
                 'Width' : 'LSTM layer = %d' %(self.width),
                 'Activation' : self.activation,
                 'Output' : self.output_units}
        return configuration

class kaux_real(tf.keras.Model):

    def __init__(self, parameter_list, name='Koopman_Aux_real'):
        super(kaux_real, self).__init__()
        self.nreal = parameter_list['num_real']
        self.units_r = parameter_list['kaux_units_real']
        self.width_r = parameter_list['kaux_width_real']
        self.activation = parameter_list['kp_activation']
        self.statet = parameter_list['stateful']

    def build(self, input_shape):
        
        self.states = []
        self.koopman_layer_real = []
        for i in range(self.width_r + 1):
            self.koopman_layer_real.append(tf.keras.layers.GRU(units= self.units_r[i], activation = self.activation, recurrent_activation = 'sigmoid', return_sequences = True, stateful = self.statet, return_state = True))

    def call(self, inputs, states):
        
        for i in range(len(self.koopman_layer_real)):
            x, state = self.koopman_layer_real[i](inputs, initial_state = states[i])
            states[i] = state
            inputs = x
        return x, states

class kaux_complex(tf.keras.Model):

    def __init__(self, parameter_list, name='Koopman_Aux_complex'):
        super(kaux_complex, self).__init__()
        self.ncomplex = parameter_list['num_complex_pairs']
        self.units_c = parameter_list['kaux_units_complex']
        self.width_c = parameter_list['kaux_width_complex'] 
        self.activation = parameter_list['kp_activation']
        self.statet = parameter_list['stateful']
    
    def build(self, input_shape):

        self.states = []
        self.koopman_layer_complex = []
        for j in range(self.width_c + 1):
            self.koopman_layer_complex.append(tf.keras.layers.GRU(units = self.units_c[j], activation = self.activation, recurrent_activation = 'sigmoid', return_sequences = True, stateful = self.statet, return_state = True))

    def call(self, inputs, states):
        inputs = tf.reduce_mean(tf.square(inputs), axis = 2, keepdims = True)
        for i in range(len(self.koopman_layer_complex)):
            y, state = self.koopman_layer_complex[i](inputs, initial_state = states[i])
            states[i] = state
            inputs = y
        return y, states

#Code for Koopman Operator Auxilary Network
class koopman_aux_net(tf.keras.Model):

    def __init__(self, kaux_real, kaux_complex, parameter_list, name='Koopman_Aux'):
        super(koopman_aux_net, self).__init__()
        self.nreal = parameter_list['num_real']
        self.ncomplex = parameter_list['num_complex_pairs']
        if self.nreal:
            self.kaux_r = kaux_real
        if self.ncomplex:
            self.kaux_c = kaux_complex
        self.output_units_complex = parameter_list['kaux_output_units_complex']
        self.output_units_real = parameter_list['kaux_output_units_real']

    def call(self, inputs, states):

        
        [r_state_s, c_state_s] = states

        # print(f'Calling Koopan_aux_net with input shape {inputs.shape}')
        input_real, input_complex = tf.split(inputs, [self.output_units_real, self.output_units_complex], axis= 2)
        # print('Shape of the real input: {}'.format(input_real.shape))
        # print('Shape of the complex input: {}'.format(input_complex.shape))

        try:
            real = []
            for i in range(self.nreal):
                x, r_state = self.kaux_r(input_real, r_state_s[i])
                r_state_s[i] = r_state
                real.append(x)
            real_tensor = tf.concat(real, 2)
        except:
            real_tensor = tf.zeros((inputs.shape[0], inputs.shape[1], 0))

        try:
            comp = []
            for j in range(self.ncomplex):
                inp_complex = input_complex[:,:,j:j+2]
                y, c_state = self.kaux_c(inp_complex, c_state_s[j])
                c_state_s[j] = c_state
                comp.append(y)
            comp_tensor = tf.concat(comp, 2)
        except:
            comp_tensor = tf.zeros((inputs.shape[0], inputs.shape[1], 0))

        ret = tf.concat([real_tensor,comp_tensor], axis=2)
        #print('Shape of the returning tensor: {}'.format(ret.shape))
        return ret, [r_state_s, c_state_s]

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

        #print(f'Calling koopman_jordan with input shape {inputs.shape}')
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

        ret = tf.concat([y_real_tensor, y_complex_tensor], axis = 2)
        #print('Shape of the returning tensor: {}'.format(ret.shape))
        return ret

class preliminary_net(tf.keras.Model):

    def __init__(self, parameter_list, encoder, decoder, k_net, k_jor, **kwargs):
        super(preliminary_net, self).__init__()
        self.encoder = encoder
        self.koopman_aux_net = k_net
        self.koopman_jordan = k_jor
        self.decoder = decoder

        self.width_r = parameter_list['kaux_width_real'] 
        self.width_c = parameter_list['kaux_width_complex'] 
        self.units_r = parameter_list['kaux_units_real']        
        self.units_c = parameter_list['kaux_units_complex'] 
        self.nreal = parameter_list['num_real']
        self.ncomplex = parameter_list['num_complex_pairs']

    def call(self, inputs):

        r = [tf.zeros((inputs.shape[0], self.units_r[s]), dtype=tf.float32) for s in range(self.width_r + 1)]
        c = [tf.zeros((inputs.shape[0], self.units_c[s]), dtype=tf.float32) for s in range(self.width_c + 1)]
        initial_stat = [[r for _ in range(self.nreal)], [c for _ in range(self.ncomplex)]]
        #This part contributes towards the (n+1)th prediction loss from nth
        k_embeddings_cur = self.encoder(inputs)

        k_omegas, _ = self.koopman_aux_net(k_embeddings_cur, initial_stat)
        k_jordan_input = tf.concat([k_omegas, k_embeddings_cur], axis= 2)
        k_jordan_output = self.koopman_jordan(k_jordan_input)

        next_state_space = self.decoder(k_jordan_output) 

        input_reconstruct = self.decoder(k_embeddings_cur)

        return next_state_space, input_reconstruct, k_embeddings_cur[:,1:,:], k_jordan_output[:,0:-1,:]

class loop_net(tf.keras.Model):

    def __init__(self, parameter_list, encoder, decoder, k_net, k_jor, **kwargs):
        super(loop_net, self).__init__()
        self.encoder = encoder
        self.koopman_aux_net = k_net
        self.koopman_jordan = k_jor
        self.decoder = decoder

        self.width_r = parameter_list['kaux_width_real'] 
        self.width_c = parameter_list['kaux_width_complex']
        self.units_r = parameter_list['kaux_units_real']        
        self.units_c = parameter_list['kaux_units_complex'] 
        self.nreal = parameter_list['num_real']
        self.ncomplex = parameter_list['num_complex_pairs']
        
        self.mth_step = parameter_list['mth_step']
    
    def build(self, input_shape):
        self.iterations = tf.constant(input_shape[1] - self.mth_step)

    def call(self, inputs):
        inputs_for_mth = inputs[:,:self.iterations,:] 
        next_state_space_mth = tf.TensorArray(tf.float32, size = self.iterations, element_shape = (inputs.shape[0], 1, inputs.shape[2]))
        k_embeddings_cur = self.encoder(inputs_for_mth)

        r = [tf.zeros((inputs.shape[0], self.units_r[s]), dtype=tf.float32) for s in range(self.width_r + 1)]
        c = [tf.zeros((inputs.shape[0], self.units_c[s]), dtype=tf.float32) for s in range(self.width_c + 1)]

        for i in tf.range(self.iterations):
            initial_stat = [[r for _ in range(self.nreal)], [c for _ in range(self.ncomplex)]]
            k_embeddings_cur_local = tf.expand_dims(k_embeddings_cur[:,i,:], axis = 1)
            k_jordan_output_local = tf.zeros_like((k_embeddings_cur_local), tf.float32)
            next_state_space_mth_local = tf.zeros((inputs.shape[0], 1, inputs.shape[2]), tf.float32)

            for j in tf.range(self.mth_step):
                k_omegas_local, states = self.koopman_aux_net(k_embeddings_cur_local, initial_stat)
                initial_stat = states
                
                k_jordan_input_local = tf.concat([k_omegas_local, k_embeddings_cur_local], axis = 2)
                k_jordan_output_local = self.koopman_jordan(k_jordan_input_local)
                k_embeddings_cur_local = k_jordan_output_local
            
            next_state_space_mth_local = self.decoder(k_jordan_output_local)
            next_state_space_mth = next_state_space_mth.write(i, next_state_space_mth_local)

        next_state_space_mth = next_state_space_mth.stack()
        next_state_space_mth = tf.squeeze(next_state_space_mth)
        next_state_space_mth = tf.transpose(next_state_space_mth, [1, 0, 2])

        return next_state_space_mth
