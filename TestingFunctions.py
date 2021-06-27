import SBPP
from SBPP.Packages import *
### A set of functions returning some physical parameters. Provided for completeness.
def beta_func(gamma):
    return np.sqrt(1-1/gamma**2)
    
def gamma_func(beta):
    return np.sqrt(1/(1-beta**2))
    
def gyration_radius_nonrel_func(mass = SBPP.Constants.e_mass,v_perp = 1,charge = SBPP.Constants.e_charge,B=1):
    return m*v_perp/(charge*B)

def gyration_init_vel_nonrel(radius=1,charge=SBPP.Constants.e_charge,mass=SBPP.Constants.e_mass,B=1):
    return radius*charge*B/mass

def gyration_radius_rel_func(mass = SBPP.Constants.e_mass,charge = SBPP.Constants.e_charge,gamma=100,v_perp=beta_func(100)*SBPP.Constants.c,B=1):
    return m*gamma*v_perp/(charge*B)
    
def cycl_freq(charge=SBPP.Constants.e_charge,field=1,mass=SBPP.Constants.e_mass):
    return charge*field/mass
    
def cycl_period(gyroradius=1,v_perp=beta_func(100)*SBPP.Constants.c):
    return 2*np.pi*gyroradius/v_perp

def Mag_Strength_rel(gyroratio=1,v_perp=beta_func(100)*SBPP.Constants.c,gamma=100,mass=SBPP.Constants.e_mass,charge=SBPP.Constants.e_charge):
    return gamma*mass*v_perp/(charge*gyroratio)
    
def mag_strength_nonRel(gyroradius = 1, v_perp = 1, mass = SBPP.Constants.e_mass, charge = SBPP.Constants.e_charge):
    return v_perp*mass/(abs(charge)*gyroradius)
    
def generate_magnetic_null_conditions(n_part,v_0):
    thetas = 2*np.pi*np.random.random(n_part)
    qs_0 = np.zeros(shape=(n_part,3))
    dqs_0 = np.zeros(shape=(n_part,3))
    for i in range(n_part):
        theta = thetas[i]
        qs_0[i] = np.array([np.cos(theta),np.sin(theta),0])
        dqs_0[i] = np.array([-v_0*np.cos(theta),-v_0*np.sin(theta),0])
    return (qs_0,dqs_0)

def generate_drift_conditions(desired_drift_velocity):
    return 0