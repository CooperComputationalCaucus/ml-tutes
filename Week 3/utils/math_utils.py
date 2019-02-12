import numpy as np
def sigmoid(x):
    '''
    Computes the sigmoid function elementwise
    
    Input
    ==============
    x : scalar, vector or martrix of values
    
    Returns
    ==============
    g : elementwise sigmoid function. Same shape as x.
    '''

    g=1/(1+np.exp(-x))

    return g