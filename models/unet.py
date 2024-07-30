import pickle 
from datetime import datetime
import sys
import numpy as np
from load_data import load_data
import os
import custom_model_elements

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
    # Fetch list of logical GPUs that have been allocated
    #  Will always be numbered 0, 1, …
    physical_devices = tf.config.get_visible_devices('GPU')
    n_physical_devices = len(physical_devices)

    # Set memory growth for each
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
	# No allocated GPUs: do not delete this case!                                                                	 
    tf.config.set_visible_devices([], 'GPU')
    
if len(sys.argv) != 2:
    raise ValueError('missing inputs')

data_file = '/scratch/bbtq/myu2/gremlin.npz'
Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
    Lat_train, Lon_train, Lat_test, Lon_test, \
    Xdata_scalar_train, Xdata_scalar_test = load_data( data_file )
nbatches_train,ny,nx,nchans = Xdata_train.shape
nbatches_test,ny,nx,nchans = Xdata_test.shape
print('ny,nx=',ny,nx)
print('nchans=',nchans)
print('nbatches train,test=',nbatches_train,nbatches_test)

# parameters
NN_string = 'UNET'  #'SEQ' or 'UNET'
activ = 'relu'
activ_last = 'linear'
activ_scalar = 'relu'
batch_size = 18
batch_step_size = 100  # 100 steps per epoch
batchnorm = True
convfilter = (3,3)
convfilter_last_layer = (1,1)
convfilter_scalar = (1,1)
data_suffix = ''
data_format = 'channels_last'
double_filters = True  #double filters for each layer
dropout = False
dropout_rate = 0.1
kernel_init = 'glorot_uniform' #'he_uniform'  ##'glorot_uniform'  # default in TF
loss = sys.argv[1]
loss_weight = (1.0, 5.0, 4.0)
machine = 'notHera'
n_conv_layers_per_decoder_layer = 1  #1=CP blocks, 2=CCP,...; for up-blocks
n_conv_layers_per_encoder_layer = 1  #1=CP blocks, 2=CCP,...
n_encoder_decoder_layers = 3 # choose integer - 0,1,2,3,...
n_filters_for_first_layer = 8
n_filters_last_layer = 1
n_filters_scalars = 16
n_scalar_layers = 2
nepochs = 250
poolfilter = (2,2)
upfilter = (2,2)
padding = 'same'
optimizer = Adam()
verby = False

modelname = '/scratch/bbtq/myu2/outputs/models/model_' + str(loss) + '_n' + str(nepochs) + '.h5'
historyname = '/scratch/bbtq/myu2/outputs/models/history_' + str(loss) + '_n' + str(nepochs) + '.bin'

# metrics
metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), custom_model_elements.gradient_mag, custom_model_elements.fft_mean, custom_model_elements.fft_99]
# if loss != 'MSE': metrics.append(tf.keras.metrics.MeanSquaredError())
# if loss != 'MAE': metrics.append(tf.keras.metrics.MeanAbsoluteError())
    
    
# loss
print('loss=',loss)
if loss == 'MSE': loss = tf.keras.losses.MeanSquaredError()
if loss == 'MAE': loss = tf.keras.losses.MeanAbsoluteError()
if loss == 'LMSE': loss = tf.keras.losses.MeanSquaredLogarithmicError()
if loss == 'PMAE': loss = tf.keras.losses.MeanAbsolutePercentageError()
if loss == 'FSS': loss = custom_model_elements.make_FSS_loss(mask_size=3)
if loss == 'Cross': loss = custom_model_elements.cross_entropy(np.full((ny,nx), True))
if loss == 'WMSE': loss = custom_model_elements.my_mean_squared_error_weighted_genexp(weight=loss_weight)
if loss == 'WMAE': loss = custom_model_elements.my_mean_absolute_error_weighted_genexp(weight=loss_weight)
if loss == 'FFT': loss = custom_model_elements.fft_loss()
if loss == 'FFT2': loss = custom_model_elements.fft2_loss()
if loss == 'Hybrid': loss = custom_model_elements.hybrid_maemse()
if loss == 'SSIM': loss = custom_model_elements.ssim()
if loss == 'MAESSIM': loss = custom_model_elements.mae_ssim()
if loss == 'GRAD': loss = custom_model_elements.gradient_loss()

stime = datetime.now()
print('\nstart', stime)
##########################################################################################
# model creation
if NN_string == 'UNET':
    IS_UNET = True
else:
    IS_UNET = False
n_filters = n_filters_for_first_layer
input_layer = Input(shape=(ny,nx,nchans))
x = input_layer
skip = []

### contracting path (encoder layers)
for i_encode_decoder_layer in range( n_encoder_decoder_layers ):
    print('Add encoder layer #' + repr(i_encode_decoder_layer) )
    if IS_UNET:
        skip.append(x)  # push current x on top of stack
        if verby:
            print(x.shape)
    for i in range(n_conv_layers_per_encoder_layer): #add conv layer
        x = Conv2D(n_filters,convfilter,activation=activ,\
            padding=padding,kernel_initializer=kernel_init,data_format=data_format)(x)
        if verby:
            print(x.shape)
        if batchnorm:
            x = BatchNormalization()(x)
            if verby:
                print(x.shape)
    x = MaxPooling2D(poolfilter,padding=padding,data_format=data_format)(x)
    if verby:
        print(x.shape)
    if dropout:
        x = Dropout(dropout_rate)(x)
    if double_filters:
        n_filters = n_filters * 2 # double for NEXT layer
        
### expanding path (decoder layers)
for i_encode_decoder_layer in range( n_encoder_decoder_layers ):
    print('Add decoder layer #' + repr(i_encode_decoder_layer) )

    # This was moved up to make endcoder and decoder symmetric.
    if double_filters:
        n_filters = n_filters // 2 # halve for NEXT layer
    
    for i in range(n_conv_layers_per_decoder_layer): #add conv layer
        # Switched from Conv2DTranspose to Conv2D.  Same functionality, but easier to visualize filters.
        x = Conv2D(n_filters,convfilter,activation=activ,\
            padding=padding,kernel_initializer=kernel_init,data_format=data_format)(x)
        if verby:
            print(x.shape)
        if batchnorm:
            x = BatchNormalization()(x)
            if verby:
                print(x.shape)
    x = UpSampling2D(upfilter,data_format=data_format)(x)
    if verby:
        print(x.shape)
    if IS_UNET:
        x = Concatenate()([x,skip.pop()]) # pop top element
    if dropout:
        x = Dropout(dropout_rate)(x)
    
if IS_UNET:
    # One additional (3x3) conv layer to properly incorporate newly
    # added channels at previous concatenate step
    x = Conv2D(n_filters,convfilter,activation=activ,\
        padding=padding,kernel_initializer=kernel_init,data_format=data_format)(x)
    if verby:
        print(x.shape)
        
# last layer: 2D convolution with (1x1) just to merge the channels
x = Conv2D(n_filters_last_layer,convfilter_last_layer,\
    activation=activ_last,padding=padding,\
    kernel_initializer=kernel_init,data_format=data_format)(x)
x = Reshape((ny,nx))(x)

model = Model(inputs=input_layer, outputs=x)

if IS_UNET:
    print('Unet !!!!')
else:
    print('SEQ !!!!')
    
print(model.summary())

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

########################################
# training
history = model.fit(Xdata_train, Ydata_train, epochs=nepochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(Xdata_test,Ydata_test))

etime = datetime.now()
print('end', etime)
print('Time ellapsed for training', (etime-stime).total_seconds(), 'seconds')

##################
# save model
print('writing model to file: ' + modelname)
model.save(modelname)

print('writing history to file: ' + historyname)
with open(historyname, 'wb') as f:
    pickle.dump({'history':history.history, 'epoch':history.epoch}, f)
    
print('Training: done')
print('end training=',datetime.now())