from SBPP import *
import SBPP.Constants
@njit
def E_array(E_function,q,t,n_part,fparams=SBPP.Constants.fparams0):
    '''This retuns a E_field, an (n_part,3) object where E_field[i] is the 3-vector containing the electric 
    field at the position of particle i at time t'''
    E_field = np.zeros(shape=(n_part,3))
    for i in range(n_part):
        E_field[i] = E_function(q[i],t,fparams)
    return E_field

@njit
def B_array(B_function,q,t,n_part,fparams=SBPP.Constants.fparams0):
    '''Identical to E_array but for the magnetic field'''
    B_field = np.zeros(shape=(n_part,3))
    for i in range(n_part):
        #print(i)
        #print(q[i])
        B_field[i] = B_function(q[i],t,fparams)
    return B_field
#Some field functions used for tests

@njit
def E_func_zero(q,t,fparams):
    E = np.array([0.0,0.0,0.0])
    return E

@njit
def B_func_zero(q,t,fparams):
    B = np.array([0,0,0])
    return B
@njit 
def E_func_uniform_arbitrary(q,t,fparams):
    Ex=fparams[0]
    Ey=fparams[1]
    Ez=fparams[2]
    E = np.array([Ex,Ey,Ez])
    return E
@njit
def B_func_uniform_arbitrary(q,t,fparams):
    Bx = fparams[3]
    By = fparams[4]
    Bz = fparams[5]
    B = np.array([Bx,By,Bz])
    return B
    
@njit
def B_func_Magnetic_Mirror(q,t,fparams):
    B0 = fparams[1]
    L = fparams[2]
    B = np.array([-q[0]*B0*q[2]/L**2,-q[1]*B0*q[2]/L**2,B0*(1+q[2]**2/L**2)])
    return B
@njit
def B_func_electromagnet(q,t,fparams):
    B0x = fparams[0]
    B0y = fparams[1]
    B0z = fparams[2]
    t_on = fparams[3]
    t_off = fparams[4]
    if (t>=t_on and t<=t_off):
        B = np.array([B0x,B0y,B0z])
    else:
        B = np.array([0.0,0.0,0.0])
    return B

@njit
def B_func_Null(q,t,fparams):
    #B0= 10**(-6)
    B0 = fparams[1]
    L = fparams[2]
    B = B0*np.array([q[1]/L,q[0]/L,0])
    return B
@njit
def B_func_Dipolar(q,t,fparams):
    M = fparams[3]
    r = q[0]**2+q[1]**2+q[2]**2
    B = M/r**(5/2)*np.array([3*q[2]*q[0],3*q[2]*q[1],(2*q[2]**2-q[0]**2-q[1]**2)])
    return B
@njit
def B_func_linear_grad(q,t,fparams):
    Bxx,Bxy,Bxz,Byx,Byy,Byz,Bzx,Bzy,Bzz = fparams
    B = np.array([Bxx*q[0]+Bxy*q[1]+Bxz*q[2],Byx*q[0]+Byy*q[1]+Byz*q[2],Bzx*q[0]+Bzy*q[1]+Bzz*q[2]])
    return B
@njit 
def B_func_rad_grad(q,t,fparams):
    B0 = fparams[0]
    B = np.array([0.0,0.0,B0*(q[0]**2+q[1]**2)**0.5])
    return B
