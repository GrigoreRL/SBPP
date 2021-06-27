#from numba import jit,njit 
from SBPP import *

### This contains a set of wrapper functions to obtain full trajectories using the SBPP pushers.

### Non-relativistic syncronized integrator. Main integrator to use!
@njit
def Integrator_Synched_nonRel(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=False,gamma_specified=0,output_Freq = 1):
'''This integrator takes initial conditions are returns a full integrated trajectory using the non-relativistic synchronized pusher. Input:
q0: initial position state. ndarray of shape (n_particles,3)
dq0: initial velocity state. ndarray of shape (n_particles,3)
E_func: electric field function. Must take 3 arguments and return a 3-component array with the field. By default it gets fed q,t and a parameter iterable
B_func: same as E_func but for the magnetic field.
masses: ndarray with n_particles components, corresponding to the particle masses.
charges: ndarray with n_particles components, corresponding to the particle charges.
dt: time-step, float.
tf: final desired time; float.
fparams1: parameter tuple to feed into the field functions.
debug: set debug mode
gamma_specified: not used; do not set.
output_frequency: sets the frequency at which the output is saved. Default is at every step.
'''
    t= 0
    qs = []
    ts = []
    dqs = []
    qs.append(q0)
    dqs.append(dq0)
    q_n = q0
    dq_n = dq0
    ctr = 0
    ts.append(t)
    while t<tf:
        ctr+=1
        q_n,dq_n = Boris_Push_Synchronized(q_n,dq_n,E_func,B_func,charges,masses,
                                                  dt,t,fparams=fparams1,gamma_specified=gamma_specified,debug=debug)
        if(ctr==output_Freq):
            qs.append(q_n)
            dqs.append(dq_n)
            ctr = 0
        t+=dt
        ts.append(t)
    return (qs,dqs,ts)
    
### Synchronized relativistic integrator. Not really intended for use, but functional. 
@njit
def Integrator_Synched(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=False,gamma_specified = 0,
output_Freq = 1,c_value = SBPP.Constants.c):
    t= 0
    qs = []
    dqs = []
    gammas=[]
    ts=[]
    ts.append(t)
    qs.append(q0)
    dqs.append(dq0)
    gammas.append(np.sqrt(1/(1-(np.linalg.norm(dq0)/c_value)**2)))
    q_n = q0
    dq_n = dq0
    ctr = 0
    while t<tf:
        ctr+=1
        q_n,dq_n,gamma_n =Boris_Push_Relativistic_Synced(q_n,dq_n,E_func,B_func,charges,masses,dt,t,fparams=fparams1,
        gamma_specified=gamma_specified,c_value = c_value)
        if ctr == output_Freq:
            qs.append(q_n)
            dqs.append(dq_n)
            gammas.append(gamma_n)
            ctr = 0
        t+=dt
        ts.append(t)
    return (qs,dqs,ts,gammas)

###########################################################################
######## Functions intended for use end here. Proceed with caution! #######
###########################################################################
@njit
def Integrator_nonRel(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=False,output_Freq = 1):
    t = 0
    qs=[]
    q0,dqm1b2 = jumpstart_push_nonRel(q0,dq0,E_func,B_func,masses,charges,dt,0,fparams=fparams1)
    q_n = q0
    #print(q_n)
    #print(dqm1b2)
    dq_n = dqm1b2
    qs.append(q_n)
    ctr = 0
    while t < tf:
        q_n, dq_n = Boris_Push(q_n,dq_n,E_func,B_func,charges,masses,dt,t,fparams=fparams1,debug = debug)
        #print(dq_n)
        if ctr == output_Freq:
            qs.append(q_n)
            ctr = 0
        t+=dt
    #print(t)
    return (qs,dq_n)
@njit
def Integrator_Rel(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=False,output_Freq = 1):
    t = 0
    qs=[]
    q0,dqm1b2 = jumpstart_push_Rel(q0,dq0,E_func=E_func_zero,B_func=B_func_Uniform_z,mass=masses,charge=charges,dt=dt,t=0,fparams=fparams1)
    q_n = q0
    #print(q_n)
    #print(dqm1b2)
    dq_n = dqm1b2
    qs.append(q_n)
    ctr = 0
    while t < tf:
        q_n, dq_n = Boris_Push_Rel(q_n,dq_n,E_func,B_func,charges,masses,dt,t,fparams=fparams1,debug = debug)
        #print(dq_n)
        if ctr == output_Freq:
            qs.append(q_n)
            ctr = 0
        t+=dt
    #print(t)
    return (qs,dq_n)


### Do not use. Attempted vectorized integrator. Actually really slow.
    '''
def Integrator_Synched_nonRel_Experimental(q0,dq0,E_func,B_func,masses,charges,dt,tf,fparams1,debug=False,gamma_specified=0,output_Freq = 1):
    t= 0
    qs = []
    dqs = []
    qs.append(q0)
    dqs.append(dq0)
    q_n = q0
    dq_n = dq0
    ctr = 0
    while t<tf:
        ctr+=1
        q_n,dq_n = Boris_Push_Synchronized_Superfast(q_n,dq_n,E_func,B_func,charges,masses,
                                                  dt,t,fparams=fparams1,gamma_specified=gamma_specified)
        if(ctr==output_Freq):
            qs.append(q_n)
            dqs.append(dq_n)
            ctr = 0
        t+=dt
    return (qs,dqs)
    '''