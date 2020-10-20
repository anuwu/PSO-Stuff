import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

class ChaosGenerator () :
    """
    Base class for the chaotic generator
    Contains functions for generating chaotic numbers and subsequently
    evolving the states of the internal generators
    """

    def getGen (shape, gentype) :
        """Returns a generator of the given shape and underlying map"""

        return (lambda s : lambda i : ChaosGenerator.cgen[gentype](s).chaosPoints(i))(shape)

    def __init__ (self, oshape, gshape=None, cascade=True, gens=2) :
        """
        Child classes use this constructor to initialise essential parameters
        and the internal generators
            oshape  - Shape that object owner uses
            gshape  - Internal shape (per generator) as the chaotic map/flow can
                   can be muti-dimensional
            cascade - If cascade=False, then each point in the (Np, D) matrix
                   evolves independently of other points according to the map.
                   For the CPSO, this amounts to having a certain correlation
                   between the random numbers r1, r2 per iteration of the CPSO
                   - If cascade=True, then each of the Np particles is connected
                   to the previous one via the chaotic map. Every dimension is
                   independent of the other, however!
            gens    - Number of independent internal chaotic generators. Two by
                   default for chaotic pso
        """

        self.oshape = oshape

        ######################################################################
        # (Np, D, cdims) --> (D, cdims)
        # where 'cdims' is the number of dimensions of the chaotic map/flow
        #
        # NOTE - By default, if map is single dimensional, then the last shape
        # dimension (of 1) is omitted
        ######################################################################
        self.gshape = (lambda s: s[1:] if cascade else s)(oshape if gshape is None else gshape)
        self.cascade = cascade
        self.gens = gens

        # Creating the list of generators with shape (gens, Np, D, cdims)
        self.cgens = np.array([
            np.random.random_sample(self.gshape)
            for i in range(gens)
        ])


    def getCgens (self) :
        """ Returns a copy of the internal generators """
        return np.copy (self.cgens)

    def chaosPoints (self, gno=0) :
        """
        Returns numbers based on the underlying chaotic map/flow and depending
        on the value of gno
            gno - If ==0 means to evolve all generators and return them as a matrix of
                shape (gens, Np, D)
                - If !=0 means to evolve a particular generator (indexed from 1) rand
                return a matrix of shape (Np, D)
        """

        if gno :
            if self.cascade :
                # Evolve per particle
                return np.array ([
                    self.evolve(gno-1) for i in range(self.oshape[0])
                ])
            else :
                return self.evolve(gno-1)
        else :
            # Evolve per generator (independent of 'cascade') --> Recursive call
            return np.array ([
                self.chaosPoints(i+1) for i in range(self.gens)
            ])

class Logistic (ChaosGenerator) :
    """
    Logistic map --> f(x) = r*x*(1-x)
    r = 4 for full chaos
    """

    def __init__ (self, oshape, cascade=True, r=4, gens=2) :
        """
        r - logistic bifurcation parameter
        Rest is defined in the parent class
        """

        ChaosGenerator.__init__(self, oshape, None, cascade, gens)
        self.r = r

    def evolve (self, gind) :
        """ Evolves according to the logistic map """

        # Copying is necessary
        x, r = self.cgens[gind], self.r
        ret = np.copy(x)

        self.cgens[gind] = r*x*(1-x)
        return ret

class InverseLE (ChaosGenerator) :
    """
        Finds a uni-dimensional map with a pre-determined lyapunov
        exponent and evolves points according to it
        Check the paper 'The problem of the inverse Lyapunov exponent and its applications'
        by Marcin Lawnik
    """

    def __invmap__ (self, eps=1e-4) :
        if self.le < np.log(2) :
            lep = lambda p : self.le + p*np.log(p) + (1-p)*np.log(1-p)
            lo, mid, hi = 0, 0.5, 1
            cmap = lambda p : lambda x : x/p if x <= p else (1-x)/(1-p)
        else :
            n = np.ceil(np.exp(self.le)).astype(np.int)
            lep = lambda p : self.le - (n-2)/n*np.log(n) + p*np.log(p) + (2/n - p)*np.log(2/n - p)
            lo, mid, hi = 0, 1/n, 2/n

            def cmap (p) :
                def _cmap(x) :
                    nx = n*x
                    nums = np.arange(0, n-2)
                    sub = nums[np.argmin(np.where(nx - nums > 0, nx - nums, n))]

                    if sub < n-3 or nx < n-2 :
                        return nx - sub
                    elif nx < n-2 + p*n : # sub == n-3
                        return (nx - (sub+1))/(n*p)
                    else :
                        return (nx - (sub+1) - n*p)/(2 - n*p)
                return _cmap


        plist = [brentq(lep, lo+eps, mid-eps), brentq(lep, mid+eps, hi-eps)]
        self.invmap = np.vectorize(cmap(plist[
            1 if np.random.rand() >= 0.5 else 0
        ]))

    def __init__ (self, oshape, cascade=True, le=1.28991999999, gens=2) :
        """ le      - The lyapunov exponent whose map has to be found """

        ChaosGenerator.__init__(self, oshape, None, cascade, gens)
        self.le = le

        if le == np.log(2) :
            mu = 0.49999
            self.invmap = lambda x : np.where(x <= mu, x/mu, (1-x)/(1-mu))
        else :
            self.__invmap__()

    def evolve (self, gind) :
        """ Evolves according to the calculated inverse map """

        # Copying is necessary
        x = self.cgens[gind]
        ret = np.copy(x)

        self.cgens[gind] = self.invmap(x)
        return ret

class Tent (ChaosGenerator) :
    """Tent map --> f(x) = 2*x , x <= 0.5 ; 2*(1-x) , x > 0.5
    mu = 0.49999 in the equivalent form for numerical stability"""

    def __init__ (self, oshape, cascade=True, mu=0.49999, gens=2) :
        """mu - Tent bifurcation paramater
        Rest is defined in the parent class"""

        ChaosGenerator.__init__(self, oshape, None, cascade, gens)
        self.mu = mu

    def evolve (self, gind) :
        """Evolves according to the tent map"""

        # Copying is necessary
        x, mu = self.cgens[gind], self.mu
        ret = np.copy(x)

        self.cgens[gind] = np.where(x <= mu, x/mu, (1-x)/(1-mu))
        return ret

class Lorenz (ChaosGenerator) :
    """
    Lorenz flow -->  xdot = sigma*(y-x)
                        ydot = x*(rho-z) - y
                        zdot = x*y - beta*z
    sigma, beta, rho = 10, 8/3, 28
    """

	# lims is a dictonary containing {(sigma, beta, rho) : limits(3,2)} pairs
    lims = {}

    def lorenz (X, t, sigma, beta, rho) :
        """ lorenz differential equation needed by scipy odeint """

        x, y, z = X
        dXdt = [sigma*(y-x), x*(rho-z) - y, x*y - beta*z]
        return dXdt

    def setLimits (params) :
        """
        No need to recalculate limits of the lorenz flow everytime for the
        same set of parameters
        """

        if params not in Lorenz.lims :
            # Argument to lambda - (Time series of lorenz flow in all three dimensions)
            Lorenz.lims[params] = (lambda s:np.array([
                [np.min(s[:,i]), np.max(s[:,i])] for i in [0, 1, 2]
            ]))\
            (odeint (Lorenz.lorenz, np.random.rand(3), np.linspace (0, 9999, 999999), args = params))



    def __init__ (self, oshape, cascade=True, params=(10, 8.0/3, 28), comp=0, h=0.01, gens=2) :
        """"
        params  - (sigma, beta, rho) of lorenz parameters
        comp    - which cdim to consider for chaotic numbers
        h       - Time step of evolution
        Rest is defined in the parent class
        """

        ChaosGenerator.__init__ (self, oshape, oshape+(3,), cascade, gens)
        self.params = params
        self.comp = comp
        self.h = h

        # Set limits if not set already
        Lorenz.setLimits (params)

        ######################################################################
        # !!!!! IDEA FOR OOP !!!!!!!
        # Introduce two subclasses - Normalised, and unnormalised
        # The unnormalised class will have normalisation functions like the one
        # below (Also seen in Henon map)
        ######################################################################

        # Per generator
        for i in range(0, self.gens) :
            # Per dimension of lorenz flow
            for j in [0, 1, 2] :
                self.cgens[i,...,j] = (lambda st,mn,mx : mn + (mx - mn)*st)\
                                    (self.cgens[i,...,j], Lorenz.lims[params][j,0], Lorenz.lims[params][j,1])
                # Argument to lambda - (ith generator jth cdim, min of jth cdim, max of jth cdim)

    def evolveT (self, gind, T=1) :
        """
        Evolves the lorenz map for T timesteps
        and sets the internal generator
        """

        for pt in np.ndindex(self.gshape[:-1]) :
        # Per index in (Np, D)
            self.cgens[gind][pt] = odeint(Lorenz.lorenz, self.cgens[gind][pt],
                                          np.arange(0,self.h*(T+1),self.h), args=self.params)[-1]

    def evolve (self, gind) :
        """
        Evolves the internal generators 1 h-timestep according to the
        Lorenz flow equations
        """

        ######################################################################
        # If the limits defined in the dict 'lims' are exceeded, then
        # corresponding chaotic points are replaced with eps or (1-eps) depending
        # on whether its exceeding below or above, respectively
        ######################################################################
        eps = 1e-5

        # Copying is not necessary as it is being scaled
        ret = (lambda n2 : np.where (n2 > 1, 1-eps, n2))(
                (lambda n1 : np.where (n1 < 0, eps, n1))(
                    (lambda st, mn, mx : (st - mn)/(mx - mn))
                    (self.cgens[gind,...,self.comp],
                     Lorenz.lims[self.params][self.comp,0],
                     Lorenz.lims[self.params][self.comp,1])
                ))

        self.evolveT (gind)
        return ret

class Henon(ChaosGenerator) :
    """
    Henon map (Simplified model of the poincare section of Lorenz model)
    (x,y) -> (1-ax^2+y, bx)
    """

    lims = {}

    def setLimits (params) :
        """ Sets the x, y limits of a run of iterates of the Henon map """

        if not params in Henon.lims :
            a, b = params
            x, y = np.random.rand(), np.random.rand()
            minx, maxx, miny, maxy = x, x, y, y

            for _ in range(999999) :
                tmp = x
                x = 1 - a*x*x + y
                y = b*tmp

                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)

            Henon.lims[params] = np.array([
                [minx, maxx], [miny, maxy]
            ])


    def __init__ (self, oshape, cascade=True, params=(1.4, 0.3), comp=0, gens=2) :
        """
        Constructor for the Henon chaotic map object
        params          - (a, b) parameters of the Henon map
        """

        ChaosGenerator.__init__ (self, oshape, oshape+(2,), cascade, gens)
        self.params = params
        self.comp = comp

        # Setting the limits for the Henon map
        Henon.setLimits(params)

        # Per generator
        for i in range(0, self.gens) :
            # Per dimension of Henon map
            for j in [0, 1] :
                self.cgens[i,...,j] = (lambda st,mn,mx : mn + (mx - mn)*st)\
                                    (self.cgens[i,...,j], Henon.lims[params][j,0], Henon.lims[params][j,1])

    def evolve (self, gind) :
        """ Evolves the Henon map by one iterate """

        # Tolerance to set back if iterate is beyond bounds
        eps = 1e-5

        # Copying is not necessary as it is being scaled
        ret = (lambda n2 : np.where (n2 > 1, 1-eps, n2))(
                (lambda n1 : np.where (n1 < 0, eps, n1))(
                    (lambda st, mn, mx : (st - mn)/(mx - mn))
                    (self.cgens[gind,...,self.comp],
                     Henon.lims[self.params][self.comp,0],
                     Henon.lims[self.params][self.comp,1])
                ))

        a, b = self.params
        x, y = np.copy(self.cgens[gind,...,0]), self.cgens[gind,...,1]
        x2 = np.square(x)
        self.cgens[gind,...,0] = 1 - a*x2 + y
        self.cgens[gind,...,1] = b*x

        return ret

class Baker (ChaosGenerator) :
    """
    Baker map -->   (2x, y/2) if 0 <= x < 1/2
                    (2-2x, 1-y/2) 1/2 <= x < 1
    """

    def __init__ (self, oshape, cascade=True, mu=0.49999, comp=0, gens=2) :

        ChaosGenerator.__init__ (self, oshape, oshape+(2,), cascade, gens)
        self.mu = mu
        self.comp = comp

    def evolve (self, gind) :
        """ Evolves one time-step according to the baker map """

        ret = np.copy(self.cgens[gind,...,self.comp])
        x, y = np.copy(self.cgens[gind,...,0]), np.copy(self.cgens[gind,...,1])
        less = x < self.mu
        more = np.invert(less)

        self.cgens[gind,less,0] = 2*x[less]
        self.cgens[gind,less,1] = y[less]/2
        self.cgens[gind,more,0] = 2 - 2*x[more]
        self.cgens[gind,more,1] = 1 - y[more]/2

        return ret

# Used by CPSO for generating swarms
ChaosGenerator.cgen = {
"log"       : Logistic,
"lorenz"    : Lorenz,
"tent"      : Tent,
"henon"     : Henon,
"baker"     : Baker,
"inverse"   : InverseLE
}
