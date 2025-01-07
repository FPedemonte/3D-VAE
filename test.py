import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm

from VAE import *
from utils import npytar, save_volume

def data_loader(fname):
    """
    Loads and preprocesses data from a .tar file containing numpy arrays
    
    Args:
        fname (str): Path to the input .tar file
        
    Returns:
        numpy.ndarray: Preprocessed data scaled to range [-1, 1]
    """
    # Initialize the tar file reader
    reader = npytar.NpyTarReader(fname)
    total = reader.length()
    print(f"Loading {total} samples from {fname}...")
    
    # Initialize empty array to store data
    xc = np.zeros((total,) + input_shape, dtype=np.float32)
    
    # Reopen the reader and load data
    reader.reopen()
    for ix, (x, name) in enumerate(tqdm(reader, total=total)):
        xc[ix] = x.astype(np.float32)
    
    # Scale data to [-1, 1] range: (x * 3) - 1
    return 3.0 * xc - 1.0

if __name__ == '__main__':
    # Initialize the VAE model
    print("Initializing model...")
    model = get_model()
    
    # Extract model components from the returned dictionary
    inputs = model['inputs']     # Input layer
    outputs = model['outputs']   # Output layer
    mu = model['mu']            # Mean vector
    sigma = model['sigma']      # Standard deviation vector
    z = model['z']             # Latent space vector
    
    # Get the encoder and decoder parts of the VAE
    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']         # Complete VAE model
    
    # Load pre-trained weights
    print("Loading weights from vae.h5...")
    vae.load_weights('vae.h5')

    # Generate reconstructions using the VAE
    print("Loading test data...")
    data_test = data_loader('datasets/shapenet10_chairs_nr.tar')

    print(f"Generating reconstructions for {len(data_test)} samples...")
    reconstructions = vae.predict(data_test, verbose=1)
    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    # Create directory for reconstructions if it doesn't exist
    if not os.path.exists('reconstructions'):
        os.makedirs('reconstructions')

    # Save each reconstruction on a separate file
    print("Saving reconstructions...")
    for i in tqdm(range(reconstructions.shape[0])):
        # Save each reconstruction using the save_output utility function
        # Parameters: reconstruction data, size (32), output directory, index
        save_volume.save_output(reconstructions[i, 0, :], 32, 'reconstructions', i)

    print("Done!")
