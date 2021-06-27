from SBPP import *
import SBPP.Constants
import math

### Main useful integrator. Synchronized non-relativistic. Will take a given state and advance to the next state.
@njit
def Boris_Push_Synchronized(q,dq,E_method,B_method,charge,mass,dt,t,debug=False,fparams=SBPP.Constants.fparams0,gamma_specified=0):
'''This function will take a given particle state at step k and advance it to step k+1. Uses a non-relativistic synchronized Boris algorithm.

 Inputs:

q: ndarray of shape (n_particles,3), containing the particle positions at the k-th step. Assumed synchronized with velocites.
dq: ndarray of shape (n_particles,3), containing the particle velocities at the k-th step.
E_method: a function which takes q, t and a tuple fparams and returns a 3-component 1-d ndarray containig the electric field at position q.
B_method: same as E_method but for the magnetic field.
charge: ndarray with n_particles components specifying the charge of each particle.
mass: ndarray with n_particles components specifying the mass of each particle.
dt: time-step to use.
t: current time. Fed into E_method and B_method; useful if they are time-dependent.
debug: enable debug mode.
fparams: parameter iterable fed into the E_method and B_method functions.
gamma_specified: not used.

Output:
Iterable (q_new,dq_new) containing the position and velocity states at the k+1th step. 
'''
    n_part = len(charge)
    dq_new = np.zeros(shape=(n_part,3))
    q_new = np.zeros(shape=(n_part,3))
    for i in range(len(charge)):
        q_half = q[i]+dq[i]*dt/2
        E_part = E_method(q_half,t+dt/2,fparams)
        B_part = B_method(q_half,t+dt/2,fparams)
        if debug:
            print("Fields follow; El. then mag.")
            print(E_part,B_part)
        qb2m = charge[i]/(2*mass[i])
        qbmEdt2 = qb2m*E_part*dt
        v_minus = dq[i]+qbmEdt2
        
        if(debug):
            print("v_minus follows")
            print(v_minus)
        t_vec = +B_part*qb2m*dt
        if debug:
            print("t_vec follows")
            print(t_vec)
        s_vec = 2*t_vec/(1+la.norm(t_vec)**2)
        
        v_minusXt = np.array([v_minus[1]*t_vec[2]-v_minus[2]*t_vec[1],
                              v_minus[2]*t_vec[0]-v_minus[0]*t_vec[2],v_minus[0]*t_vec[1]-v_minus[1]*t_vec[0]])
        
        v_prime = v_minus + v_minusXt
        
        vprimeXs = np.array([v_prime[1]*s_vec[2]-v_prime[2]*s_vec[1],
                             v_prime[2]*s_vec[0]-v_prime[0]*s_vec[2],v_prime[0]*s_vec[1]-v_prime[1]*s_vec[0]])
        if debug:
            print(vprimeXs)
        v_plus = v_minus + vprimeXs
        
        v_np1 = v_plus+qbmEdt2
        q_np1 = q_half + v_np1*dt/2
        q_new[i]=q_np1
        dq_new[i] = v_np1
        
    return (q_new,dq_new)


### Relativistic synchronized pusher. Generally same as the non-rel version, but uses the relativistic Boris algorithm.
@njit
def Boris_Push_Relativistic_Synced(q,dq,E_method,B_method,charge,mass,dt,t,
                                   fparams=SBPP.Constants.fparams0,c_value = 299792458,gamma_specified=0):       
'''This function will take a given particle state at step k and advance it to step k+1. Inputs:

q: ndarray of shape (n_particles,3), containing the particle positions at the k-th step. Assumed synchronized with velocites.
dq: ndarray of shape (n_particles,3), containing the particle velocities at the k-th step.
E_method: a function which takes q, t and a tuple fparams and returns a 3-component 1-d ndarray containig the electric field at position q.
B_method: same as E_method but for the magnetic field.
charge: ndarray with n_particles components specifying the charge of each particle.
mass: ndarray with n_particles components specifying the mass of each particle.
dt: time-step to use.
t: current time. Fed into E_method and B_method; useful if they are time-dependent.
debug: enable debug mode.
fparams: parameter iterable fed into the E_method and B_method functions.
gamma_specified: used if the electric field is vanishing to keep a predetermined gamma. Experimental. Not used if set to 0.

Output:
Iterable (q_new,dq_new) containing the position and velocity states at the k+1th step. 
'''
    c = c_value #Enables one to set c=1 for natural units; other quantities must be adjusted; use with care!
    ### Initialize container object
    n_part = len(charge) #Number of particles is the length of the charge vector
    dq_new = np.zeros(shape=(n_part,3)) #Initialize the new velocity state
    q_new = np.zeros(shape=(n_part,3)) #Initialize the new position state
    #gammas = np.zeros(shape=(n_part,1)) #Vector containing gamma factors
    
    ### Now initializing the fields
    #E = E_array(E_method,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    #B = B_array(B_method,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    #E_field_zero = ((E==0).all()==True and gamma_specified!=0)
    #print(E_field_zero)
    ### Fields computed
    ### Integrate equations of motion for each particle
    for i in range(len(charge)):
        beta_n = la.norm(dq[i])/c
        beta_n2 = la.norm(dq[i])**2/c**2
        #gamma_n = np.sqrt(1/(1-beta_n**2))
        gamma_n = 1/np.sqrt(1-beta_n2)
        #if gamma_n>1e7:
        #    break
        E_part = E_method(q[i],t,fparams)
        E_field_zero = ((E_part==0).all()==True and gamma_specified!=0)
        if E_field_zero:
            gamma_n = gamma_specified
        #print(gamma,'\n')
        #gamma = gamma_approximation(dq[i])
        #print(gamma,'\n')
        
        
        
        u_n = gamma_n*dq[i]
        q_half = q[i]+dq[i]*dt/2
        
        #Getting the fields at half-position-step
        E_part = E_method(q_half,t+dt/2,fparams)
        B_part = B_method(q_half,t+dt/2,fparams)
        #Checking for specified gamma and pure magnetic field
        E_field_zero = ((E_part==0).all()==True and gamma_specified!=0)
        
        qbmEdt2 = charge[i]/mass[i]*E_part*dt/2  
        qb2m = charge[i]/(2*mass[i])
        
        ### Compute auxiliary quantities
        u_minus = u_n + qbmEdt2
        
        #gamma_minus = np.sqrt(1+(la.norm(u_minus)/c)**2)
        gamma_minus = np.sqrt(1+(la.norm(u_minus))**2/c**2)
        if E_field_zero:
            gamma_minus = gamma_specified
        t_vec = B_part*qb2m*dt*1/gamma_minus
        s_vec = 2*t_vec/(1+la.norm(t_vec)**2)
        
        u_minusXt = np.array([u_minus[1]*t_vec[2]-u_minus[2]*t_vec[1],
                              u_minus[2]*t_vec[0]-u_minus[0]*t_vec[2],u_minus[0]*t_vec[1]-u_minus[1]*t_vec[0]])
        
        u_prime = u_minus + u_minusXt
        
        uprimeXs = np.array([u_prime[1]*s_vec[2]-u_prime[2]*s_vec[1],
                             u_prime[2]*s_vec[0]-u_prime[0]*s_vec[2],u_prime[0]*s_vec[1]-u_prime[1]*s_vec[0]])
        
        u_plus = u_minus+uprimeXs
        
        u_np1 = u_plus+qbmEdt2
        #gamma_nphalf = np.sqrt(1+(la.norm(u_plus)/c)**2)
        gamma_nphalf = np.sqrt(1+(la.norm(u_plus))**2/c**2)
        #gamma_np1 = np.sqrt(1+(la.norm(u_np1)/c)**2)
        gamma_np1 = np.sqrt(1+(la.norm(u_np1))**2/c**2)
        v_np1 = u_np1/gamma_np1
        if E_field_zero:
            gamma_np1 = gamma_specified
        dq_new[i] = v_np1
        q_new[i] = q_half+u_np1/gamma_np1 * dt/2
        ### Loop done
        t+=dt
    ### Return the result
    return (q_new,dq_new,gamma_np1)

#####################################################
### Functions intended for use end here #############
#####################################################


### Nonrelativistic integrator without built-in synching; do not use; present for completeness
@njit
def Boris_Push(q,dq,E_method,B_method,charge,mass,dt,t,
               debug=False,fparams=SBPP.Constants.fparams0):
    '''This function takes the state (x_k,v_(k-1/2)) and takes it to the state (x_(k+1),v_(k+1/2))'''
    ### Initialize container objects
    n_part = len(charge) #Number of particles is the length of the charge vector
    dq_new = np.zeros(shape=(n_part,3)) #Initialize the new velocity state
    q_new = np.zeros(shape=(n_part,3)) #Initialize the new position state
    if(debug):
        print("Initialized containers\n")
        print("Time step follows")
        print(dt)
    ### Done with initial objects
    
    ### Now initializing the fields
    E = E_array(E_method,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    B = B_array(B_method,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    ### Fields computed
    if debug:
        print("Fields Follow")
        print(E,B)
        print("Charge to mass ratios follow")
        print(charge/mass)
    
    ### Integrate equations of motion for each particle
    if debug:
        print("Integrating forward for each particle\n")
    for i in range(len(charge)):
        ### Precomputations
        qbmEdt2 = charge[i]/mass[i]*E[i]*dt/2  
        qb2m = charge[i]/(2*mass[i])
        invdt = 1/dt
        ### Precomputations done
        ### Compute auxiliary quantities
        
        v_minus = dq[i]+qbmEdt2
        if(debug):
            print("v_minus follows")
            print(v_minus)
        t_vec = +B[i]*qb2m*dt
        if debug:
            print("t_vec follows")
            print(t_vec)
        s_vec = 2*t_vec/(1+la.norm(t_vec)**2)
        
        v_minusXt = np.array([v_minus[1]*t_vec[2]-v_minus[2]*t_vec[1],
                              v_minus[2]*t_vec[0]-v_minus[0]*t_vec[2],v_minus[0]*t_vec[1]-v_minus[1]*t_vec[0]])
        
        v_prime = v_minus + v_minusXt
        
        vprimeXs = np.array([v_prime[1]*s_vec[2]-v_prime[2]*s_vec[1],
                             v_prime[2]*s_vec[0]-v_prime[0]*s_vec[2],v_prime[0]*s_vec[1]-v_prime[1]*s_vec[0]])
        if debug:
            print(vprimeXs)
        v_plus = v_minus + vprimeXs
        
        
        ### Computed auxiliary quantities
        ###Store new states
        dq_new[i] = v_plus+qbmEdt2
        q_new[i] = q[i]+dq_new[i]*dt
        ### Loop done
    if debug:
        print("Integration done")
    ### Return the result
    return (q_new,dq_new)




### Relativistic non-synchronized pusher. Don't use.
@njit
def Boris_Push_Rel(q,dq,E_method,B_method,charge,mass,dt,t,debug=False,
                   fparams=SBPP.Constants.fparams0,gamma_specified = 0):
    '''This function takes the state (x_k,v_(k-1/2)) and takes it to the state (x_(k+1),v_(k+1/2))'''
    ### Initialize container objects
    n_part = len(charge) #Number of particles is the length of the charge vector
    dq_new = np.zeros(shape=(n_part,3)) #Initialize the new velocity state
    q_new = np.zeros(shape=(n_part,3)) #Initialize the new position state
    if(debug):
        print("Initialized containers\n")
    ### Done initializing
    
    ### Now initializing the fields
    E = E_array(E_method,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    B = B_array(B_method,q,t,n_part,fparams=fparams) #Fields at each particle's position at the given time
    E_field_zero = ((E==0).all()==True and gamma_specified!=0)
    ### Fields computed
    
    ### Integrate equations of motion for each particle
    if debug:
        print("Integrating forward for each particle\n")
    for i in range(len(charge)):
        ### Precomputations
        qbmEdt2 = charge[i]/mass[i]*E[i]*dt/2  
        qb2m = charge[i]/(2*mass[i])
        invdt = 1/dt
        ### Precomputations done
        ### Compute auxiliary quantities
        if E_field_zero:
            gamma = gamma_specified
        else:
            gamma = np.sqrt(1/(1-(la.norm(dq[i])/c)**2))
            
        u_minus = gamma*dq[i]+qbmEdt2
        
        if E_field_zero:
            gamma = gamma_specified
        else:
            gamma = np.sqrt(1+(la.norm(u_minus)/c)**2)
        
        t_vec = B[i]*qb2m*dt*1/gamma
        s_vec = 2*t_vec/(1+la.norm(t_vec)**2)
        
        u_minusXt = np.array([u_minus[1]*t_vec[2]-u_minus[2]*t_vec[1],
                              u_minus[2]*t_vec[0]-u_minus[0]*t_vec[2],u_minus[0]*t_vec[1]-u_minus[1]*t_vec[0]])
        
        u_prime = u_minus + u_minusXt
        
        uprimeXs = np.array([u_prime[1]*s_vec[2]-u_prime[2]*s_vec[1],
                             u_prime[2]*s_vec[0]-u_prime[0]*s_vec[2],u_prime[0]*s_vec[1]-u_prime[1]*s_vec[0]])
        
        u_plus = u_minus + uprimeXs
        
        u_new = u_plus + qbmEdt2
        v_new = u_new/gamma
        ### Computed auxiliary quantities
        ###Store new states
        dq_new[i] = v_new
        q_new[i] = q[i]+dq_new[i]*dt
        ### Loop done
    if debug:
        print("Integration done")
    ### Return the result
    return (q_new,dq_new)
    



#@njit
#### This was an attempt to do fully vectorized computations. It is not superfast. It is not even fast. Don't use it.
'''
def Boris_Push_Synchronized_Superfast(q,dq,E_method,B_method,charge,mass,dt,t,debug=False,fparams=SBPP.Constants.fparams0,gamma_specified=0):
    n_part = len(charge)
    
    dq_new = np.zeros(shape=(n_part,3))
    q_new = np.zeros(shape=(n_part,3))
    
    q_half = q+dq*dt/2
    E = E_array(E_method,q_half,t,n_part,fparams=fparams)
    B = B_array(E_method,q_half,t,n_part,fparams=fparams)
    print(B)
    qb2m = charge/(2*mass)
    qbmEdt2 = qb2m * E * dt 
    
    v_minus = dq + qbmEdt2 
    
    t_vec = B*qb2m*dt 
    
    t_norm = np.sum(np.abs(t_vec)**2,axis=-1)**0.5
    s_vec = 2*t_vec/(1+(t_norm.reshape(-1,1))**2)
    
    v_minusXt = np.cross(v_minus,t_vec)
    
    v_prime = v_minus + v_minusXt 
    
    vprimeXs = np.cross(v_prime,s_vec)
    
    v_plus = v_minus + vprimeXs
    
    v_np1 = v_plus + qbmEdt2
    
    q_np1 = q_half +v_np1*dt/2
    
    q_new = q_np1
    dq_new = v_np1
    return (q_new,dq_new)
'''