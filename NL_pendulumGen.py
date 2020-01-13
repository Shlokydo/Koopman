#For an interactive visulaization of the Lorenz attractor
import numpy as np
from scipy import integrate

import pandas as pd
from mpl_toolkits.axisartist.axislines import SubplotZero

def nl_pendulum(N = 10, max_time = 10, delta_t= 0.02, show = False, x0 = [-3.1, 3.1], x1 = [-2 , 2]):
    
    def nl_pendulum_deriv(x_y, t0):
        """Compute the time-derivative."""
        x, y = x_y
        return [y, -np.sin(x)]
    
    x = np.resize(np.linspace(x0[0], x0[1], num = N),(N,1))[:-1]
    ra = np.random.permutation(len(x))[:int(len(x)/2)]
    np.random.shuffle(x)
    x[ra] = x[ra] * -1
    y = np.resize(np.linspace(x1[0], x1[1], num = N),(N,1))[:-1]
    y[ra] = y[ra] * -1
    x = np.insert(x, 1, y.T, axis=1)

    # Solve for the trajectories
    time = np.arange(0, max_time+delta_t, delta_t)
    
    x_t_new = []
    for x0i in x:
        potential = 0.5*(np.power(x0i[1],2)) - np.cos(x0i[0])
        if potential < 0.99:
            x_t_new.append(integrate.odeint(nl_pendulum_deriv, x0i, time))
            
    x_t_new = np.asarray(x_t_new)

    return time, x_t_new

time, x_t = nl_pendulum(N = 35000, max_time = 1, delta_t=0.02, show= False, x0= [-3.1, 0], x1 = [-2, 0])
time, x_t2 = nl_pendulum(N = 35000, max_time = 10, delta_t=0.2, show= False, x0= [-3.1, 0], x1 = [-2, 0])
print(x_t.shape)

#Code to make Panda's Dataframe out of the simulation data

def package(x_t, key, startpoints, endpoints, dname):
    dataframe = pd.DataFrame()

    x_t = x_t[startpoints:endpoints]
    m,n,r = x_t.shape
    out_arr = np.column_stack((np.repeat(np.arange(m),n),x_t.reshape(m*n,-1)))
    out_df = pd.DataFrame(out_arr, columns=['Iteration', 'X', 'Y'])
    dataframe = dataframe.append(out_df, ignore_index= True)

    #Writing the dataframe to a global Dataset HDF5 file
    dataframe.to_hdf(dname, key=key, mode='a')

package(x_t[:1024*4*5.1], 'nl_pendulum_tenth', 0, 1024*4*5, 'Dataset.h5')
package(x_t[:1024*4*5.1], 'nl_pendulum_tenth_test', 1024*4*5, -1, 'Dataset.h5')
package(x_t2[:1024*4*5.1], 'nl_pendulum', 0, 1024*4*5, 'Dataset.h5')
package(x_t2[:1024*4*5.1], 'nl_pendulum_test', 1024*4*5, -1, 'Dataset.h5')