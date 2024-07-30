import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def compute_beta_alpha(nsteps, beta_start, beta_end, gamma_start=0, gamma_end=0.1):
    '''
    Create the beta, alpha and gamma sequences.
    Element 0 is closest to the true image; element NSTEPS-1 is closest to the
       completely noised image
       
    '''
    beta = np.arange(beta_start, beta_end, (beta_end-beta_start)/nsteps)
    gamma = np.arange(gamma_start, gamma_end, (gamma_end-gamma_start)/nsteps)
    alpha = np.cumprod(1-beta)

    return beta, alpha, gamma

def convert_image(I):
    '''
    Convert an image from a form where the pixel values are nominally in a +/-1 range
    into a range of 0...1
    '''
    I = I/2.0 + 0.5
    I = np.maximum(I, 0.0)
    I = np.minimum(I, 1.0)
    return I


'''

Position Encoder Layer

Creates an Attention-Like Positional encoding.  The input tensor
then selects which rows to return.

Source: Hands-On Machine Learning, p 558

'''
class PositionEncoder(keras.layers.Layer):
    def __init__(self, max_steps:int, max_dims:int, 
                 dtype=tf.float32, **kwargs):
        '''
        Constructor

        :param max_steps: the number of tokens in the sequence
        :param max_dims: the length of the vector used to encode position
                    (must match the token encoding length if "add")
        :param dtype: The type used for encoding of position
        '''
        # Call superclass constructor
        super().__init__(dtype=dtype, **kwargs)

        # Deal with odd lengths
        if max_dims % 2 == 1: max_dims += 1

        # Create the positional representation
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((max_steps, max_dims))
        pos_emb[:, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[:, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T

        # Save the state
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
        
    def call(self, indices):
        '''
        This method is what implements the object "callable" property.

        Determines how the input tensor is translated into the output tensor.

        :param inputs: TF Tensor that indicates which rows
        :return: TF Tensor
        '''
        return tf.gather_nd(self.positional_embedding, tf.cast(indices, dtype='int32'))

    def embedding(self):
        return self.positional_embedding

def create_time_encoding(time, size):
    '''
    Create a time encoding for the given time and size
    Parameters:
        time: tf.Tensor, size: int
    Returns:
        time_encoded: tf.Tensor
    '''
    time_encoded = PositionEncoder(20, size)(time)
    time_encoded = tf.expand_dims(tf.expand_dims(time_encoded, axis=-1), axis=-1)
    time_encoded = tf.tile(time_encoded, [1, 1, size, 1])
    return time_encoded

def create_generator(image_size, n_times, nchannels, conv_layers, p_spatial_dropout, 
                                  lambda_l2, isUnet, padding, 
                                  conv_activation, n_layers, pool, lrate,
                                  loss='binary_crossentropy',
                                  metrics=[tf.metrics.SparseCategoricalAccuracy()]):
    '''
    Create a U-Net generator network with the given parameters
    Parameters:
        image_size: int, n_times: int, nchannels: int, conv_layers: list of dicts, p_spatial_dropout: float,
        lambda_l2: float, isUnet: bool, n_classes: int, padding: string, conv_activation: string,
        n_layers: int, pool: int, lrate: float, loss: string, metrics: list of tf.keras.metrics
    Returns:
        model: tf.keras.Model
    '''
    # regularization
    if lambda_l2 != None:
        l_reg = tf.keras.regularizers.L2(l2=lambda_l2)
    else:
        l_reg = None
        
    # model creation
    # inputs
    image_input = layers.Input(shape=(image_size, image_size, 3), name='image_input')
    time_input = layers.Input(shape=(1,), name='time_input') #TODO not hardcode 8
    label_input = layers.Input(shape=(image_size, image_size, 7), name='label_input')

    # time encoding
    times = [create_time_encoding(time_input, image_size//(2**i)) for i in range(n_times-1,-1,-1)]

    input_layer = [image_input, time_input, label_input]
    x = layers.Concatenate()([label_input, image_input])
    
    if isUnet:
        skip = []

    # encoding layers
    for layer in range(n_layers):
        if isUnet:
            skip.append(x)
        
        for layer in conv_layers:
            x = layers.Conv2D(layer['filters'], layer['kernel_size'], activation=conv_activation, padding=padding, kernel_regularizer=l_reg)(x)
                        
            if layer['batch_normalization']:
                x = layers.BatchNormalization()(x)
            if p_spatial_dropout != None:
                x = layers.SpatialDropout2D(p_spatial_dropout)(x)

        x = layers.MaxPooling2D(pool, padding=padding)(x)

    x = layers.Concatenate()([x, times.pop(0)])
    # decoding layers
    for layer in range(n_layers):
        for layer in conv_layers:
            x = layers.Conv2D(layer['filters'], layer['kernel_size'], activation=conv_activation, padding=padding, kernel_regularizer=l_reg)(x)
                        
            if layer['batch_normalization']:
                x = layers.BatchNormalization()(x)
            if p_spatial_dropout != None:
                x = layers.SpatialDropout2D(p_spatial_dropout)(x)

        x = layers.UpSampling2D(pool)(x)
        if times != []:
            x = layers.Concatenate()([x, times.pop(0)])
        if isUnet:
            x = layers.Concatenate()([x, skip.pop()])

    # that final jene c'est quoi
    for layer in conv_layers:
        x = layers.Conv2D(layer['filters'], layer['kernel_size'], activation=conv_activation, padding=padding, kernel_regularizer=l_reg)(x)
                        
        if layer['batch_normalization']:
            x = layers.BatchNormalization()(x)
        if p_spatial_dropout != None:
            x = layers.SpatialDropout2D(p_spatial_dropout)(x)

    # final softmax layer
    x = layers.Conv2D(nchannels, 1, activation='sigmoid')(x)

    
    # model compilation
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate), loss=loss, metrics=metrics)
    
    return model