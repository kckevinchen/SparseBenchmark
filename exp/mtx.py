import scipy.io as sio
import numpy as np
import os

def load_from_mtx(mtx_path):
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    
    """
    matrix = sio.mmread(mtx_path).astype(np.float32).todense()
    return matrix