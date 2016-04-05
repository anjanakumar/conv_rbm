import numpy as np

def extend_image(image, steps):
    # Required number of rows
    required_rows = steps-(np.mod(image.shape[0], steps)) 
    # Required number of columns
    required_cols = steps -(np.mod(image.shape[1], steps)) 
    # Concatenate original array with rows filled with zeros 
    partial = np.concatenate((image, np.zeros((required_rows,image.shape[1])))) 
    # Concatenate partial array with columns filled with zeros
    extended = np.concatenate((partial, np.zeros((partial.shape[0], required_cols))), axis=1) 
    return extended,required_rows,required_cols
