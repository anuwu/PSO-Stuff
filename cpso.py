#################################################################
# Should probably structure the variables among classes and
# sub-classes efficiently later. Also, I haven't checked whether
# the chaotic search part in PWLC_PSO and HECS_PSO work.
#################################################################

import numpy as np
from collections import deque

import pso_util as pu
import chaosGen as cg

norm = lambda x : np.sqrt(np.sum(np.square(x)))
norms = lambda X : np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))

class PSO () :
    """
    Base class for a variety of PSO optimizers
    """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.01, cache=False) :
        """
        Constructor of base PSO optimizer -
            obj         - Objective function to minimize
            llim        - Left limits in each dimension
            rlim        - Right limits in each dimension
            vrat        - Velocity limit ratio
        """

        self.obj = obj
        self.objkey = lambda x : self.obj(x.reshape(1, -1))[0]
        self.llim = llim
        self.rlim = rlim
        self.Np = Np
        self.vrat = vrat
        self.cache = cache

        self.vmax = vrat*(rlim - llim).reshape(1,-1)

    # Consider a better design for this later
    def __setcache__ (self) :
        """
        Sets all the caches to the empty list if cache parameter
        in initialisation is true
        """

        ######################################################################
        # Caches to hold optimization iterations for the last optimization
        # performed
        # Contains  - position
        #           - velocity
        #           - momentum
        #           - pbest
        #           - gbest
        ######################################################################

        if self.cache :
            (self.pcache,
            self.vcache,
            self.pbcache,
            self.gbcache,
            self.r1cache,
            self.r2cache) = [], [], [], [], [], []
        else :
            (self.pcache,
            self.vcache,
            self.pbcache,
            self.gbcache,
            self.r1cache,
            self.r2cache) = None, None, None, None, None, None

    def initParticles (self) :
        """
        Initialises particle position and velocity.
        Called at beginning of optimize()
        """

        # The only way to find the dimension of the swarm
        self.D = len(self.llim)

        self.particles = np.array([l + (r-l)*np.random.rand(self.D, self.Np)[ind] \
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

        self.velocity = np.array([self.vrat*(r-l)*(2*np.random.rand(self.D, self.Np)[ind] - 1)\
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

    def __optiminit__ (self) :
        self.__setcache__()
        self.initParticles()
        pbest = np.copy(self.particles)
        gbest = np.copy(min(pbest, key = self.objkey))

        self.appendCache(self.particles, self.velocity, pbest, gbest)
        return pbest, gbest

    def optimize (self, w=0.7, c1=2, c2=2, max_iters=10000, vtol=1e-4) :
        """ Optimization loop of plain PSO """

        pbest, gbest = self.__optiminit__()

        i = 0
        while True :
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key=self.objkey)
            self.appendCache (self.particles, self.velocity, pbest, gbest, r1, r2)

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > 100 and (np.abs(self.velocity) < vtol).all() :
                break

        return gbest

    def appendCache (self, p, v, pb, gb, r1=None, r2=None) :
        """ Called every iteration of optimize() """

        if self.cache :
            self.pcache.append(np.copy(p))
            self.vcache.append(np.copy(v))
            self.pbcache.append(np.copy(pb))
            self.gbcache.append(np.copy(gb))
            if r1 is not None : self.r1cache.append(np.copy(r1))
            if r2 is not None : self.r2cache.append(np.copy(r2))

    def numpifyCache (self) :
        """ Sets cache list in numpy format. Called before exit of optimize() """

        if self.cache :
            self.pcache = np.array(self.pcache)
            self.vcache = np.array(self.vcache)
            self.pbcache = np.array(self.pbcache)
            self.gbcache = np.array(self.gbcache)
            self.r1cache = np.array(self.r1cache)
            self.r2cache = np.array(self.r2cache)

class ChaoticAdaswarm (PSO) :
    """
    AdaSwarm with random number generators replaced with chaotic generators

    Name    - AdaSwarm: A Novel PSO optimization Method for the Mathematical Equivalence of Error Gradients
    Author  - Rohan et. al.
    Link    - https://arxiv.org/abs/2006.09875
    """

    def __setcache__ (self) :
        super().__setcache__()
        self.mcache = [] if self.cache else None

    def __init__ (self, obj, llim, rlim, Np, initgen, randgen, vrat=0.01, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator

        The rest are defined in the base class PSO()
        """

        super().__init__(obj, llim, rlim, Np, vrat, cache)
        self.initgen = initgen
        self.randgen = randgen
        self.__setcache__()

    def appendCache (self, p, v, m, pb, gb, r1=None, r2=None) :
        super().appendCache(p, v, pb, gb, r1, r2)
        self.mcache.append(np.copy(m))

    def numpifyCache (self) :
        """ Sets cache list in numpy format. Called before exit of optimize() """

        super().numpifyCache()
        if self.cache :
            self.mcache = np.array(self.mcache)


    def initParticles (self) :
        # The only way to find the dimension of the swarm
        self.D = len(self.llim)

        # Uses the first internal generator --> initgen(1)
        self.particles = np.array([l + (r-l)*self.initgen(1).transpose()[ind] \
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

        # Uses the second internal generator --> initgen(2)
        self.velocity = np.array([self.vrat*(r-l)*(2*self.initgen(2).transpose()[ind] - 1)\
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

    def __optiminit__ (self) :
        pbest, gbest = super.__optiminit__()
        momentum = np.zeros(shape=self.particles.shape)
        return momentum, pbest, gbest

    def optimize (self, get_grad=False, c1=2, c2=2, alpha=1.2, beta=0.9, max_iters=10000, vtol=1e-4) :
        """
        Performs the PSO optimization loop
        Arguments are default PSO parameters
        Returns the optimum found, and lambda function for approximate gradient
        """

        momentum, pbest, gbest = self.__optiminit__()

        i = 0
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1, r2 = self.randgen(1), self.randgen(2)

            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Check function docstring for details
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = self.objkey)

            # Append to cache after updating particles, velocities, pbest and gbest
            self.appendCache (self.particles, self.velocity, momentum, pbest, gbest, r1, r2)

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > 100 and (np.abs(self.velocity) < vtol).all() :
                break

        # Convert cache list to numpy ndarray
        self.numpifyCache ()

        return gbest, lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta)) \
            if get_grad else gbest

    def replay (self, seed, c1=2, c2=2, alpha=1.2, beta=0.9) :
        """
        Given a pre-determined sequence of r1, r2 and a starting
        position, velocity and momentum, replays the PSO trajectory

        Typically, this is meant to be used after perturbing the starting
        position slightly.
        """

        part, vel, mom, pb, gb, r1s, r2s = seed
        seedcopy = ()
        for s in seed :
            seedcopy += (np.copy(s), )

        (part,
        vel,
        mom,
        pb,
        gb,
        r1s,
        r2s) = seedcopy

        (pcache,
        vcache,
        mcache,
        pbcache,
        gbcache) = [part], [vel], [mom], [pb], [gb]

        for r1, r2 in zip(r1s, r2s) :
            mom = beta*mom + (1-beta)*vel
            vel = mom + c1*r1*(pb - part) + c2*r2*(gb - part)
            vel = pu.vclip(vel, self.vmax)
            part, vel = pu.ipcd(part, vel, self.llim, self.rlim, alpha)

            less = self.obj(part) < self.obj(pb)
            pb[less] = part[less]
            gb = min(pb , key = lambda x : self.obj(x.reshape(1,-1))[0])

            pcache.append(part)
            vcache.append(vel)
            mcache.append(mom)
            pbcache.append(pb)
            gbcache.append(gb)

        return np.array(pcache), np.array(vcache), np.array(mcache), np.array(pbcache), np.array(gbcache)

class HECS_PSO (PSO) :
    """
    Name    - A Hybrid Particle Swarm Algorithm with Embedded Chaotic Search
    Author  - Meng et. al.
    Link    - https://ieeexplore.ieee.org/document/1460442
    """

    def __init__ (self, obj, llim, rlim, Np, stag_tol=1e-3, Nc=6, Gmax=500, rrat=0.2, vrat=0.01, cache=False) :
        """
        Constructor for the hybrid embedded chaotic search PSO optimizer -
            stag_tol        - Stagnation tolerance for kicking in chaotic search
            Nc              - Number of iterations to check for stagnation
            Gmax            - Maximum iterations in the chaotic search
            rrat            - Carrier wave radius in chaotic search

        Rest are defined in the base class
        """

        super().__init__(obj, llim, rlim, Np, vrat, cache)
        self.stag_tol = stag_tol
        self.Nc = Nc
        self.Gmax = Gmax
        self.rrat = rrat

    def __optiminit__ (self) :
        pbest, gbest = super().__optiminit__()
        fitness_q = deque(maxlen=self.Nc)
        fitness_q.append(self.obj(self.particles))
        self.appendCache(self.particles, self.velocity, pbest, gbest)

        return fitness_q, pbest, gbest

    def optimize (self, w=0.7, c1=2, c2=2, alpha=1.2, max_iters=10000, vtol=1e-2) :
        """ Runs the PSO loop """

        fitness_q, pbest, gbest = self.__optiminit__()

        i = -1
        while True :
            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > 100 and (np.abs(self.velocity) < vtol).all() :
                break

            # Chaotic search
            if i >= self.Nc :
                fits_ps = np.array(fitness_q).transpose()
                for j, fits_p in enumerate(fits_ps) :
                    if ((fits_p - self.obj(gbest.reshape(1, -1)))/fits_p < self.stag_tol).all() :
                        cgen = cg.Logistic((self.Gmax, self.D), gens=1)
                        cp = self.particles[j] + self.rrat*(self.rlim - self.llim)*(2*cgen.chaosPoints(1) - 1)
                        obj_cp = np.where(np.logical_and(self.llim.reshape(1, -1) <= cp, cp <= self.rlim.reshape(1, -1)), self.obj(cp), np.inf)
                        gbest_p = np.argmin(obj_cp).flatten()

                        if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(self.particles[j]) :
                            self.velocity[j] = self.particles[j] - cp[gbest_p]

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Check function docstring for details
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest, key = self.objkey)

            # Append to cache after updating particles, velocities, pbest and gbest
            fitness_q.append(self.obj(self.particles))

            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.appendCache (self.particles, self.velocity, pbest, gbest, r1, r2)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

        return gbest

class PWLC_PSO (PSO) :
    """
    Name    - An improved particle swarm optimization algorithm combined with piecewise linear chaotic map
    Author  - Xiang et. al.
    Link    - https://www.sciencedirect.com/science/article/abs/pii/S0096300307002081
    """

    def __init__ (self, obj, llim, rlim, Np, mu=0.7, rrat=0.8, rho=0.9, vrat=0.01, cache=False) :
        """
        Constructor for the hybrid embedded chaotic search PSO optimizer -
            stag_tol        - Stagnation tolerance for kicking in chaotic search
            Nc              - Number of iterations to check for stagnation
            Gmax            - Maximum iterations in the chaotic search
            rrat            - Carrier wave radius in chaotic search

        Rest are defined in the base class
        """

        super().__init__(obj, llim, rlim, Np, vrat, cache)
        self.mu = mu
        self.rrat = rrat
        self.rho = rho

    def __optiminit__ (self) :
        pbest, gbest = super().__optiminit__()
        self.appendCache(self.particles, self.velocity, pbest, gbest)

        return pbest, gbest

    def optimize (self, w=0.7, c1=2, c2=2, alpha=1.2, max_chaos_iters=500, max_pso_iters=10000, vtol=1e-4) :
        """ Optimization loop of plain PSO """

        pbest, gbest = self.__optiminit__()

        i = 0
        cgen = cg.Tent((max_chaos_iters, self.D), mu=self.mu, gens=1)
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Check function docstring for details
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest_ind = np.argmin(self.obj(pbest)).flatten()
            gbest = pbest[gbest_ind]

            cp = gbest + self.rrat*(self.rlim - self.llim)*(2*cgen.chaosPoints(1) - 1)
            obj_cp = np.where(np.logical_and(self.llim.reshape(1,-1) <= cp, cp <= self.rlim.reshape(1,-1)), self.obj(cp), np.inf)
            gbest_p = np.argmin(obj_cp).flatten()

            if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(pbest[gbest_ind]) :
                self.velocity[gbest_ind] = 0
                pbest[gbest_ind] = cp[gbest_p]
                self.particles[gbest_ind] = pbest[gbest_ind]

            # Append to cache after updating particles, velocities, pbest and gbest
            self.appendCache (self.particles, self.velocity, pbest, gbest, r1, r2)

            self.rrat *= self.rho
            i += 1
            print("\r{}".format(i), end="")
            if i == max_pso_iters or \
            i > 100 and (np.abs(self.velocity) < vtol).all() :
                break

class GB_PSO (PSO) :
    """
    Name        - An Adaptive Velocity Particle Swarm Optimization for High-Dimensional Function Optimization
    Author      - Arasomwan et. al.
    Link        - https://ieeexplore.ieee.org/document/6557850
    """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.01, cache=False) :
        super().__init__(obj, llim, rlim, Np, vrat, cache)

    def __setcache__ (self) :
        """
        Sets all the caches to the empty list if cache parameter
        in initialisation is true
        """

        ######################################################################
        # Caches to hold optimization iterations for the last optimization
        # performed
        # Contains  - position
        #           - velocity
        #           - momentum
        #           - pbest
        #           - gbest
        ######################################################################

        if self.cache :
            (self.pcache,
            self.vcache,
            self.gbcache,
            self.r1cache,
            self.r2cache) = [], [], [], [], []
        else :
            (self.pcache,
            self.vcache,
            self.gbcache,
            self.r1cache,
            self.r2cache) = None, None, None, None, None

    def appendCache (self, p, v, gb, r1=None, r2=None) :
        """ Called every iteration of optimize() """

        if self.cache :
            self.pcache.append(np.copy(p))
            self.vcache.append(np.copy(v))
            self.gbcache.append(np.copy(gb))
            if r1 is not None : self.r1cache.append(np.copy(r1))
            if r2 is not None : self.r2cache.append(np.copy(r2))

    def __optiminit__ (self) :
        self.__setcache__()
        self.initParticles()
        gbest = np.copy(min(self.particles, key = self.objkey))
        self.appendCache(self.particles, self.velocity, gbest)
        return gbest

    def optimize (self, c1=2, c2=2, alpha=1.2, max_iters=10000, vtol=1e-4) :
        """ Optimization loop of plain PSO """

        gbest = self.__optiminit__()

        i = 0
        while True :
            max_dist = norm(gbest)
            p_dists = norms(self.particles - gbest)
            self.velocity = (gbest - self.particles)*p_dists/max_dist

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Check function docstring for details
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            self.appendCache (self.particles, self.velocity, gbest)
            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > 100 and (np.abs(self.velocity) < vtol).all() :
                break

        return gbest
