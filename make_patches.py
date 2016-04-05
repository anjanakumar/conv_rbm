import matplotlib.pyplot as p
import matplotlib.cm as cm
from skimage.util.shape import view_as_windows
from PIL import Image
import numpy as np

def make_patch(data,patch_size,step):
    ''' Function to make patches of images '''
    ''' The function do the normalization too'''
    ''' The func was setup to make (100,100) batches'''
    img = data
    arr = np.array(data, dtype = np.float32)
    #arr = (arr-0)/(255-0)
    batch = (patch_size,patch_size)
    patch = view_as_windows(arr, batch,step)

    return patch

