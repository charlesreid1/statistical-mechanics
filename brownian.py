import numpy as np

def brownian(delta, dt, N, d):
    """Returns a d-dimensional Brownian walk 
    with length scale delta
    and random step size drawn from 
    zero-mean normal distribution 
    with standard deviation delta^2*dt.
    
    N steps in d dimensional space.

    Parameters:
        
        delta : length scale of diffusion step
        dt : time scale of diffusion step
        N : number of diffusion steps
        d : dimensionality of Brownian walk
    """
    mean = 0.0
    stdev = np.sqrt(delta*delta*dt)
    
    # Get N-1 random steps
    r = np.random.normal(loc=mean, scale=stdev, size=(N-1,d))
    
    # Make sure the first location is 0,0
    r2 = np.zeros((N, d))
    r2[1:N,:] = r[0:(N-1),:]
    
    return r2.cumsum(axis=0)
