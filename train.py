import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from VAE import * # Import VAE model and components
from utils import npytar # Import utility for loading .tar files

learning_rate_1 = 0.0001 # Initial learning rate
learning_rate_2 = 0.005 # Learning rate after epoch 1
momentum = 0.9 # Momentum for SGD optimizer
batch_size = 10 # Batch size for training
epoch_num = 150 # Number of epochs to train for

def data_loader(fname):
    """
    Loads and preprocesses data from a .tar file containing numpy arrays
    
    Args:
        fname (str): Path to the input .tar file
        
    Returns:
        numpy.ndarray: Preprocessed data scaled to range [-1, 1]
    """
    reader = npytar.NpyTarReader(fname)
    xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
    reader.reopen()
    for ix, (x, name) in enumerate(reader):
        xc[ix] = x.astype(np.float32)
    return 3.0 * xc - 1.0

def weighted_binary_crossentropy(target, output):
    """
    Custom loss function that applies different weights to positive and negative samples
    Positive samples (1s) are weighted 98%, negative samples (0s) are weighted 2%
    
    Args:
        target: True values
        output: Predicted values
        
    Returns:
        Weighted binary cross-entropy loss
    """
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def learning_rate_scheduler(epoch, lr):
    """
    Callback function to adjust learning rate during training
    
    Args:
        epoch: Current epoch number
        lr: Current learning rate
        
    Returns:
        New learning rate
    """
    if epoch >= 1:
        lr = learning_rate_2
    return lr

if __name__ == '__main__':
    """
    Main function to train the VAE model
    """
    model = get_model() # Get (initialize) the VAE model

    # Extract model components from the returned dictionary
    inputs = model['inputs'] # Input layer
    outputs = model['outputs'] # Output layer
    mu = model['mu'] # Mean vector
    sigma = model['sigma'] # Standard deviation vector
    z = model['z'] # Latent space vector

    encoder = model['encoder'] # Encoder part of the VAE
    decoder = model['decoder'] # Decoder part of the VAE

    # Comment out all plot_model calls
    # plot_model(encoder, to_file = 'vae_encoder.pdf', show_shapes = True)
    # plot_model(decoder, to_file = 'vae_decoder.pdf', show_shapes = True)
    # plot_model(vae, to_file = 'vae.pdf', show_shapes = True)

    vae = model['vae'] # Complete VAE model

    # Add custom loss function to the model
    voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32') # Clip values to avoid log(0)
    vae.add_loss(voxel_loss)

    # Configure optimizer with initial learning rate
    sgd = SGD(learning_rate=learning_rate_1, momentum=momentum, nesterov=True)
    vae.compile(optimizer=sgd)

    # Load training data
    data_train = data_loader('datasets/shapenet10_chairs_nr.tar')

    # Train the VAE model
    vae.fit(
        x=data_train,
        y=data_train,
        epochs=epoch_num,
        batch_size=batch_size,
        validation_split=0.1, # 10% of data for validation
        callbacks=[LearningRateScheduler(learning_rate_scheduler)] # Adjust learning rate during training
    )

    # Save the trained model weights
    vae.save_weights('vae.h5')
