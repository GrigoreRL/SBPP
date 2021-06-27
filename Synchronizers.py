from SBPP import *
import SBPP.Constants 
### A set of synchronizer functions. This approach has been abandoned and is obsolete so these functions should not see any use.
@njit
def sync_outputs(q,dq,E,B,charge,mass,dt,t,fparams=SBPP.Constants.fparams0):
    '''This function applies a half-Euler step to synch the velocities to the positions in time; the half-Euler step
    introduces a small bounded error in the phase space plots, since we do not use the synched velocities for anything other 
    than output at the specified time-step;
    the function takes the state (x_k,v_(k-1/2)) and returns the output (x_k,v_k) which can be used for plots.
    '''
    n_part = len(charge)
    E = E_array(E,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    B = B_array(B,q,t,n_part,fparams=fparams)
    dq_synched = np.zeros(shape=(n_part,3))
    for i in range(n_part):
        qbm = charge[i]/(mass[i])
        v_wedge_B =  np.array([dq[i,1]*B[i,2]-dq[i,2]*B[i,1],dq[i,2]*B[i,0]-dq[i,0]*B[i,2],dq[i,0]*B[i,1]-dq[i,1]*B[i,0]])
        qbmE = qbm*E[i]
        qbmvxB = qbm*v_wedge_B
        dq_synched[i] = dq[i]+1/2*dt*(qbmE+qbmvxB)
    return dq_synched
    
    
@njit
def sync_outputs_Relativistic(q,dq,E,B,charge,mass,dt,t,fparams=SBPP.Constants.fparams0,c_value = 299792458):
    c_value = c
    '''This function applies a half-Euler step to synch the velocities to the positions in time; the half-Euler step
    introduces a small bounded error in the phase space plots, since we do not use the synched velocities for anything other 
    than output at the specified time-step;
    the function takes the state (x_k,v_(k-1/2)) and returns the output (x_k,v_k) which can be used for plots.
    '''
    n_part = len(charge)
    E = E_array(E,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    B = B_array(B,q,t,n_part,fparams=fparams)
    dq_synched = np.zeros(shape=(n_part,3))
    for i in range(n_part):
        qbm = charge[i]/(mass[i])
        v_wedge_B =  np.array([dq[i,1]*B[i,2]-dq[i,2]*B[i,1],dq[i,2]*B[i,0]-dq[i,0]*B[i,2],dq[i,0]*B[i,1]-dq[i,1]*B[i,0]])
        qbmE = qbm*E[i]
        qbmvxB = qbm*v_wedge_B
        beta = la.norm(dq[i])/c
        gamma = np.sqrt(1/(1-beta**2))
        u_old = dq[i]*gamma
        u_jump = u_old + 1/2*dt*(qbmE+qbmvxB)
        gamma = np.sqrt(1+(la.norm(u_jump)/c)**2)
        v_jump = u_jump/gamma
        dq_synched[i] = v_jump
    return dq_synched