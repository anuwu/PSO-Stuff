import numpy as np
from scipy.integrate import odeint

class ChaosGen () :    
    def __init__ (self, *args) :
        if len(args) > 0 :
            self.vec = np.random.random_sample (args[0])
            self.shape = args[0]
        else :
            self.vec = None
            self.shape = None
            
    def getIterate (self) :
        return self.vec
                 
    def chaosPoints (self, steps=1) :
        ret = self.vec
        self.evolveSteps (steps)
        return ret
    
    def evolve (self) :
        ''' Defined in the specific chaotic classes '''
        pass
    
    def evolveSteps (self, steps) :
        for i in range(0, steps) :
            self.evolve ()

class Logistic (ChaosGen) :
    def __init__ (self, shape, r=4) :
        ChaosGen.__init__(self, shape)
        self.r = r

    def evolve (self) :
        self.vec = self.r*self.vec*(1-self.vec)

class Tent (ChaosGen) :
    def __init__ (self, shape, mu=0.49999) :
        ChaosGen.__init__(self, shape)
        self.mu = mu
        
    def evolve (self) :
        self.vec = np.where (self.vec <= self.mu, self.vec/self.mu, (1-self.vec)/(1-self.mu))

class Lorenz (ChaosGen) :
    def lorenz (X, t, sigma, beta, rho) :
        x, y, z = X
        return [sigma*(y-x), x*(rho-z) - y, x*y - beta*z]
    
    def __init__ (self, shape, params=(10, 8.0/3, 28)) :
        ChaosGen.__init__(self)
        self.shape = shape
        self.params = params
        
        sol = odeint (Lorenz.lorenz, np.random.rand(3), np.linspace (0, 9999, 999999), args = params)
        self.lims = np.array ([[np.min(sol[:,i]), np.max(sol[:,i])] for i in range(0, 3)])
        
        self.state = np.empty (shape + (3,))
        self.vec = np.random.random_sample (shape)
        
        scaleState = (lambda mn, mx, st=np.random.random_sample (shape) : mn + (mx - mn)*st)
        self.state[...,0] = scaleState (self.lims[0,0], self.lims[0,1], self.vec)
        self.state[...,1] = scaleState (self.lims[1,0], self.lims[1,1])
        self.state[...,2] = scaleState (self.lims[2,0], self.lims[2,1])
        
        
    def evolveSteps (self, steps) :
        eps = 1e-5
        self.state = np.array ([\
                odeint(Lorenz.lorenz, self.state[pt], np.arange(0,0.01*(steps+1),0.01), args=self.params)[-1]\
                for pt in np.ndindex(self.shape)\
                ]).reshape(self.shape + (3,))
        
        self.vec = (lambda n2 : np.where (n2 > 1, 1-eps, n2))(\
                (lambda n1 : np.where (n1 < 0, eps, n1))(\
                (lambda st, mn, mx : (st - mn)/(mx - mn))(self.state[...,0], self.lims[0][0], self.lims[0][1])\
                ))
        
        
    def evolve (self) :
        self.evolveSteps (1)
