import numpy as np
import matplotlib.pyplot as plt

def display_data(X,order='C'):
    '''
    Creates plot of 100 greyscale images from flattened vectors
    
    Inputs
    ================
    X : (m,n) flattened array of greyscale pixel values
    order : {‘C’, ‘F’, ‘A’} 
    
    Returns
    ================
    fig : matplotlib figure to show
    '''
    m,n = X.shape
    m=76
    display_rows = int(np.floor(np.sqrt(m)));
    display_cols = int(np.ceil(m / display_rows));
    
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols,
                            gridspec_kw={'wspace':.1, 'hspace':.1},
                            figsize=(6,6))
    idx=0
    for i in range(display_rows):
        for j in range(display_cols):
            ax=axes[i,j]
            ax.imshow(X[idx].reshape((20, 20),order=order),cmap='gray',aspect='auto')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            idx+=1
    return fig
