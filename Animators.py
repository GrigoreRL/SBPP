from SBPP import *
import SBPP.Constants
### Some experimental and not-used classes for particle animation. (INCREDIBLY early) Precursors to the SBTV application.
### These are provided for academic completeness ONLY and are not intended to be used in any way, shape or form.
class ParticleAnimator:
    def __init__(self,q_start,dq_start,charges,masses,integrator,colors=(),debug=False):
        self.debug = debug
        if debug:
            print("Initializing class\n")
        self.n_part = len(charges)
        self.q = q_start
        self.dq = dq_start
        self.q_history = []
        self.dq_history = []
        self.dq_history.append(self.dq)
        self.q_history.append(self.q)
        self.charges=charges
        self.masses = masses
        self.t = 0
        #self.fig = plt.figure()
        #self.frame = self.fig.add_subplot(111,projection='3d')
        #self.frame.set_xlabel("x[m]")
        #self.frame.set_ylabel("y[m]")
        #self.frame.set_zlabel("z[m]")
        #self.xdata = []
        #self.ydata= []
        #self.zdata = []
        #for i in range(self.n_part):
        #    self.xdata.append(self.q[i,0])
        #    self.ydata.append(self.q[i,1])
        #    self.zdata.append(self.q[i,2])
        #self.ln, = self.frame.plot(self.xdata,self.ydata,self.zdata,'ro')
        self.integrator = integrator
        if debug:
            print("q:{},dq:{},charges:{},masses:{}".format(self.q,self.dq,self.charges,self.masses))
            print("Initialized\n")
    def plot(self):
        self.fig = plt.figure()
        self.frame = self.fig.add_subplot(111,projection='3d')
        self.frame.set_xlabel("x[m]")
        self.frame.set_ylabel("y[m]")
        self.frame.set_zlabel("z[m]")
        self.xdata = []
        self.ydata= []
        self.zdata = []
        for i in range(self.n_part):
            self.xdata.append(self.q[i,0])
            self.ydata.append(self.q[i,1])
            self.zdata.append(self.q[i,2])
        self.ln, = self.frame.plot(self.xdata,self.ydata,self.zdata,'ro')
    def advance(self,E_method,B_method,dt,fparams,tf=None):
        if tf == None:
            self.q,self.dq = self.integrator(self.q,self.dq,E_method,B_method,self.charges,self.masses,dt,self.t,fparams=fparams)
            self.t+= dt
        else:
            while self.t<tf:
                self.q,self.dq = self.integrator(self.q,self.dq,E_method,B_method,self.charges,self.masses,dt,self.t,fparams=fparams)
                self.t+=dt
    @njit
    def advance_and_store(self,E_method,B_method,dt,fparams,tf=0,store_freq = 1,show_time=False):
        if tf == 0:
            self.q,self.dq = self.integrator(self.q,self.dq,E_method,B_method,self.charges,self.masses,dt,self.t,fparams=fparams)
            self.q_history.append(self.q)
            self.dq_history.append(self.dq)
            self.t+= dt
        else:
            time_start = time.time()
            ctr = 0
            while self.t<tf:
                self.q,self.dq = self.integrator(self.q,self.dq,E_method,B_method,self.charges,self.masses,dt,self.t,fparams=fparams)
                if ctr == store_freq:
                    self.q_history.append(self.q)
                    self.dq_history.append(self.dq)
                    ctr = 0
                self.t+=dt
                ctr +=1
            time_end = time.time()
            print(time_end-time_start)
    def update_animation(self,i,E_method,B_method,dt,fparams):
        if debug:
            print("Called\n")
        advance(E_method,B_method,dt,fparams)
        if debug:
            print("q:{}\n".format(self.q))
        self.ln, = self.frame.plot([self.q[i,0]],[self.q[i,1]],[self.q[i,2]],'ro')
        return self.ln
    
    def animate(self,E_method,B_method,dt,fparams):
        anim = FuncAnimation(self.fig,self.update_animation,fargs = (E_method,B_method,dt,fparams))
        
class ParticleAnimatorWorking:
    def __init__(self,q_hist,dq_hist, dt, charges,masses,debug = False, colors = ()):
        self.debug = debug
        if debug:
            print("Initializing class\n")
        self.n_part = len(charges)
        self.q = np.array(q_hist)
        self.dq = np.array(dq_hist)
        self.charges=charges
        self.masses = masses
        self.ts = np.array([i*dt for i in range(len(self.q[:,0,0]))])
        self.iters = np.array([i for i in range(len(self.q[:,0,0]))])
        #self.fig = plt.figure()
        #self.frame = self.fig.add_subplot(111,projection='3d')
        #self.frame.set_xlabel("x[m]")
        #self.frame.set_ylabel("y[m]")
        #self.frame.set_zlabel("z[m]")
        #self.xdata = []
        #self.ydata= []
        #self.zdata = []
        #for i in range(self.n_part):
        #    self.xdata.append(self.q[i,0])
        #    self.ydata.append(self.q[i,1])
        #    self.zdata.append(self.q[i,2])
        #self.ln, = self.frame.plot(self.xdata,self.ydata,self.zdata,'ro')
        if debug:
            print("q:{},dq:{},charges:{},masses:{}".format(self.q,self.dq,self.charges,self.masses))
            print("Initialized\n")
    def initialize_fig(self,xlim=(-1,1),ylim=(-1,1),zlim=(-1,1)):
        self.fig = plt.figure()
        self.frame = self.fig.add_subplot(111,projection='3d')
        self.frame.set_xlabel("x[m]")
        self.frame.set_ylabel("y[m]")
        self.frame.set_zlabel("z[m]")
        self.frame.set_xlim(xlim[0],xlim[1])
        self.frame.set_ylim(ylim[0],ylim[1])
        self.frame.set_zlim(zlim[0],zlim[1])
        self.xdata = []
        self.ydata= []
        self.zdata = []
        self.ln, = self.frame.plot(self.xdata,self.ydata,self.zdata,'ro')
        return (self.fig,self.frame,self.ln)
    def update_animation(self,iteration):
        if self.debug:
            print("Called\n")
        if self.debug:
            print("q:{}\n".format(self.q))
        if iteration<10:
            self.xdata.append(self.q[iteration,0,0])
            self.ydata.append(self.q[iteration,0,1])
            self.zdata.append(self.q[iteration,0,2])
        else:
            self.xdata = self.q[iteration-10:iteration,0,0]
            self.ydata = self.q[iteration-10:iteration,0,1]
            self.zdata = self.q[iteration-10:iteration,0,2]
        
        self.ln.set_data_3d(self.xdata,self.ydata,self.zdata)
        return self.ln
    
    def animate(self):
        anim = FuncAnimation(self.fig,self.update_animation,frames=self.iters)
        return anim

#class AnimationPlayerWindow(FuncAnimation):
#    def __init__(self,Animator):
    