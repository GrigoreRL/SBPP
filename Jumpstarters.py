from SBPP import *

### A set of "jumpstart" functions. This approach has been abandoned and these are considered obsolete.

@njit
def jumpstart_push_nonRel(q0,dq0,E_func,B_func,charge,mass,dt,t,fparams=SBPP.Constants.fparams0):
    '''This function implements the jumpstart procedure for the Boris algorithm; it takes
    (x_0,v_0) to (x_0,v_(-1/2))
    '''
    ### Initialization
    q_old = q0
    n_part = len(charge)
    dq_jumpstarted = np.zeros(shape=(n_part,3))
    
    ### Field computation
    E = E_array(E_func,q0,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    B = B_array(B_func,q0,t,n_part,fparams=fparams)
    ### Fields done
    
    ### Jumpstart for each particle
    for i in range(n_part):
       #Get v_(-1/2)
        qbm = -charge[i]/(mass[i])
        v_wedge_B = np.array([dq0[i,1]*B[i,2]-dq0[i,2]*B[i,1],dq0[i,2]*B[i,0]-dq0[i,0]*B[i,2],dq0[i,0]*B[i,1]-dq0[i,1]*B[i,0]])
        qbmE = qbm*E[i]
        qbmvxB = qbm*v_wedge_B
        dq_jumpstarted[i] = dq0[i]+1/2*dt*(qbmE+qbmvxB)
    return (q_old,dq_jumpstarted)
    
    
@njit
def jumpstart_push_Rel(q0,dq0,E_func=E_func_zero,B_func=B_func_zero,charge=np.array([SBPP.Constants.e_charge]),
                       mass = np.array([SBPP.Constants.e_mass]),dt = 1e-9,t=0,fparams=SBPP.Constants.fparams0):
    ### Initialization
    q_old = q0
    n_part = len(charge)
    q_jumpstarted=np.zeros(shape=(n_part,3))
    dq_jumpstarted = np.zeros(shape=(n_part,3))
    
    ### Field computation
    E = E_array(E_func,q0,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    B = B_array(B_func,q0,t,n_part,fparams=fparams)
    ### Fields done
    
    ### Jumpstart for each particle
    for i in range(n_part):
        qbm = -charge[i]/(mass[i])
        v_wedge_B = np.array([dq0[i,1]*B[i,2]-dq0[i,2]*B[i,1],dq0[i,2]*B[i,0]-dq0[i,0]*B[i,2],dq0[i,0]*B[i,1]-dq0[i,1]*B[i,0]])
        qbmE = qbm*E[i]
        qbmvxB = qbm*v_wedge_B
        beta = la.norm(dq0[i])/c
        gamma = np.sqrt(1/(1-beta**2))
        u_old = dq0[i]*gamma
        u_jump = u_old + 1/2*dt*(qbmE+qbmvxB)
        gamma = np.sqrt(1+(la.norm(u_jump)/c)**2)
        v_jump = u_jump/gamma
        dq_jumpstarted[i] = v_jump
    return (q_old,dq_jumpstarted)