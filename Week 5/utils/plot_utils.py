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
    m=100
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

def display_2d_binary(X,labels):
    '''
    Creates a plot of a 2-d dataset with binary labels

    Inputs
    ================
    X : (m,2) array of 2d feature vectors
    labels : (m,1) like binary labeling of 0 or 1
    
    Returns
    ================
    fig : matplotlib figure to show
    '''
    m,n = X.shape
    pos = [idx for (idx,val) in enumerate(labels) if val==1]
    neg = [idx for (idx,val) in enumerate(labels) if val==0]
    
    fig, ax = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'wspace':.1, 'hspace':.1},
                            figsize=(6,6))
    ax.plot(X[pos,0],X[pos,1],'kx',MarkerSize=8,label='Positive')
    ax.plot(X[neg,0],X[neg,1],'ko',MarkerFaceColor='yellow',MarkerSize=8,label='Negative')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    return fig

def display_boundary_linear(X,labels,clf):
    '''
    Creates a plot of a 2-d dataset with binary labels

    Inputs
    ================
    X : (m,2) array of 2d feature vectors
    labels : (m,1) like binary labeling of 0 or 1
    clf : sklearn classifier
    
    Returns
    ================
    fig : matplotlib figure to show
    '''
    
    m,n = X.shape
    pos = [idx for (idx,val) in enumerate(labels) if val==1]
    neg = [idx for (idx,val) in enumerate(labels) if val==0]
    
    fig, ax = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'wspace':.1, 'hspace':.1},
                            figsize=(6,6))
    ax.plot(X[pos,0],X[pos,1],'kx',MarkerSize=8,label='Positive')
    ax.plot(X[neg,0],X[neg,1],'ko',MarkerFaceColor='yellow',MarkerSize=8,label='Negative')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    w=clf.coef_[0]
    b=clf.intercept_
    xlin = np.array([np.min(X[:,0]),np.max(X[:,0])])
    ylin = -(w[0]*xlin+b)/w[1]
    ax.plot(xlin,ylin,'b-',label="Decision Boundary")
    ax.legend()
    return fig

def display_boundary_nonlinear(X,labels,clf):
    '''
    Creates a plot of a 2-d dataset with binary labels

    Inputs
    ================
    X : (m,2) array of 2d feature vectors
    labels : (m,1) like binary labeling of 0 or 1
    clf : sklearn classifier
    
    Returns
    ================
    fig : matplotlib figure to show
    '''
    xx,yy = np.mgrid[np.min(X[:,0]):np.max(X[:,0]):(np.max(X[:,0])-np.min(X[:,0]))/100,
                    np.min(X[:,1]):np.max(X[:,1]):(np.max(X[:,1])-np.min(X[:,1]))/100]

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, Z, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    
    ax.contour(xx, yy, Z, levels=[0.5],vmin=0, vmax=1,colors='green')
    
    ax.scatter(X[:,0], X[:, 1], c=labels[:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    
    ax.set(xlabel="$X_1$", ylabel="$X_2$",
           xlim = [np.min(X[:,0]),np.max(X[:,0])],
           ylim = [np.min(X[:,1]),np.max(X[:,1])])
    return f