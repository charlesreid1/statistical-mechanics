import numpy as np
from scipy.integrate import odeint

def get_lorenz_solution(in_0, tmax, nt, args_tuple):
    t = np.linspace(0, tmax, nt)
    soln = odeint(lorenz, in_0, t, args=args_tuple).T
    return t, soln

def lorenz(in_, t, sigma, b, r):
    """Evaluates the RHS of the 3 
    Lorenz attractor differential equations.

    in_ : initial vector of [x_0, y_0, z_0]
    t : time vector (not used, but present for odeint() call)
    sigma : numerical parameter 1
    b :     numerical parameter 2
    r :     numerical parameter 3
    """
    x = in_[0]
    y = in_[1]
    z = in_[2]
    return [sigma*(y-x),
            r*x - y - x*z,
            x*y - b*z]

