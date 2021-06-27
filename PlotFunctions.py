from SBPP import *
import SBPP.Constants
#import numpy as np
#import numpy.linalg as la
#import scipy as sp
#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import time
#from matplotlib.animation import FuncAnimation
#from numba import jit,njit,prange
from numba import jit,njit,prange

# A set of quick-n-dirty plotting functions.
#@njit 
def Plot_Results_3d(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,integrator,debug=False,show_time=False,
                    gamma_specified=0,output=False,output_Freq = 1,returnFigs=False):
                    '''This function will display a 3D plot of the trajectory corresponding to the input parameters.
                    '''
    if show_time:
        time0 = time.time()
    qs,dqs,ts = integrator(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=debug,gamma_specified=gamma_specified,output_Freq = output_Freq)
    if show_time:
        time1 = time.time()
        print(time1-time0,'\n')
    n_part = len(masses)
    qs = np.array(qs)
    dqs = np.array(dqs)
    fig = plt.figure()
    frame = fig.add_subplot(111,projection='3d')
    #print(qs)
    for part in range(n_part):
        xs = qs[:,part,0]
        ys = qs[:,part,1]
        zs = qs[:,part,2]
        traj,=frame.plot(xs,ys,zs)
    frame.set_xlabel("x[m]")
    frame.set_ylabel("y[m]")
    frame.set_zlabel("z[m]")
    plt.show()
    if output:
        if returnFigs:
            return (qs,dqs,(fig,frame,traj))
        else:
            return (qs,dqs,ts)
    else:
        if returnFigs:
            return ((fig,frame,traj))
        else:
            return 1
def Plot_Results_2d(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,integrator,debug=False,show_time=False,
                    gamma_specified=0,output=False,output_Freq = 1):
                    '''This function will display 3 2D plots of the trajectory corresponding to the input parameters.
                    '''
    if show_time:
        time0 = time.time()
    qs,dqs,ts = integrator(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=debug,gamma_specified=gamma_specified,output_Freq = output_Freq)
    if show_time:
        time1 = time.time()
        print(time1-time0,'\n')
    n_part = len(masses)
    qs = np.array(qs)
    dqs = np.array(dqs)
    figxy = plt.figure()
    framexy = figxy.add_subplot(1,1,1)
    figxz = plt.figure()
    framexz = figxz.add_subplot(1,1,1)
    figyz = plt.figure()
    frameyz = figyz.add_subplot(1,1,1)
    #print(qs)
    for part in range(n_part):
        xs = qs[:,part,0]
        ys = qs[:,part,1]
        zs = qs[:,part,2]
        framexy.plot(xs,ys)
        framexz.plot(xs,zs)
        frameyz.plot(ys,zs)
        
    framexy.set_xlabel("x[m]")
    framexy.set_ylabel("y[m]")
    framexz.set_xlabel("x[m]")
    framexz.set_ylabel("z[m]")
    frameyz.set_xlabel("y[m]")
    frameyz.set_ylabel("z[m]")
    plt.show()
    if output:
        return (qs,dqs,ts)
    else:
        return 0