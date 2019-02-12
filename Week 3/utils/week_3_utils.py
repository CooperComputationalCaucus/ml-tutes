import scipy.io as sio
import numpy as np

def load_dataset():
    matlab_data = sio.loadmat('../data/ex3data1.mat')
    X = matlab_data['X']
    X = X.reshape((X.shape[0],20,20),order='F')
    y = matlab_data['y']
    y[y==10] = 0
    
    np.random.seed(4)
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    
    split = int(np.floor(y.shape[0]*0.8))
    train_X = X[:split,...]
    train_y = y[:split]
    test_X = X[split:,...]
    test_y = y[split:]
    return train_X,train_y,test_X, test_y

def indicies_to_one_hot(data,n_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]