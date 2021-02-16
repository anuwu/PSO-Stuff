import numpy as np
import pso
import chaosGen as cg
from collections import deque

class EMPSO (pso.PSO) :
    """
    AdaSwarm with random number generators replaced with chaotic generators

    Name    - AdaSwarm: A Novel PSO optimization Method for the Mathematical Equivalence of Error Gradients
    Author  - Rohan et. al.
    Link    - https://arxiv.org/abs/2006.09875
    """

    def get_chaotic_swarm (swarm_args) :
        """ Returns a warm of particles with specified initialiser and underlying dynamics """

        def ret_swarm (obj, llim, rlim, Np) :
            D = len(llim)

            # Returns the chaotic generator
            get_gen = lambda dic : \
                    None if dic is None else \
                    lambda x : cg.cgen[dic['name']](*(((Np, D), ) + dic['args'])).chaosPoints(x)

            initgen, randgen = get_gen(swarm_args['init_cmap']), get_gen(swarm_args['dyn_cmap'])
            return EMPSO(obj, llim, rlim, Np, initgen, randgen)

        return ret_swarm

    def get_plain_swarm () :
        """ Returns a plain swarm """
        return lambda obj, llim, rlim, Np : EMPSO(obj, llim, rlim, Np)

    def __setcache__ (self) :
        """
        Sets all the caches to the empty list if cache parameter
        in initialisation is true
        """

        ######################################################################
        # Caches to hold optimization iterations for the last optimization
        # performed. Contains the following -
        #   - position
        #   - velocity
        #   - momentum
        #   - pbest
        #   - gbest
        #   - r1
        #   - r2
        ######################################################################

        if self.cache :
            (self.pcache,
            self.vcache,
            self.mcache,
            self.pbcache,
            self.gbcache,
            self.r1cache,
            self.r2cache) = [], [], [], [], [], [], []
        else :
            (self.pcache,
            self.vcache,
            self.mcache,
            self.pbcache,
            self.gbcache,
            self.r1cache,
            self.r2cache) = None, None, None, None, None, None, None

    def __init__ (self, obj, llim, rlim, Np, initgen=None, randgen=None, vrat=0.1, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator

        The rest are defined in the base class PSO()
        """

        super().__init__(obj, llim, rlim, Np, vrat)

        # Chaotic generators for the swarm initialiser
        self.initgen = (lambda x : np.random.rand(self.Np, self.D)) \
                    if initgen is None else initgen
        self.randgen = (lambda x : np.random.rand(self.Np, self.D)) \
                    if randgen is None else randgen

        # PSO progress cache for replaying history (used in calculating LE of trajectory)
        self.cache = cache
        self.__setcache__()

    def __str__ (self) :
        """ Optimizer descriptor """
        return "EMPSO"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        momentum = np.zeros_like(self.particles)
        self.__setcache__()
        self.appendCache(self.particles, self.velocity, momentum, pbest, gbest)
        return momentum, pbest, gbest

    def optimize (self, c1=1, c2=1, alpha=1.2, beta=0.9,
                max_iters=10000, tol=1e-2,
                print_iters=False) :
        """
        Performs the PSO optimization loop
        Arguments are default PSO parameters
        Returns the optimum found, and lambda function for approximate gradient
        """

        momentum, pbest, gbest = self._optim_init()

        i = 0
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1, r2 = self.randgen(1), self.randgen(2)

            # Momentum and velocity update
            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = self.objkey)
            self.conv_curve.append(self.objkey(gbest))

            # Append to cache after updating particles, velocities, pbest and gbest
            self.appendCache (self.particles, self.velocity, momentum, pbest, gbest, r1, r2)

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        # Convert cache list to numpy ndarray
        self.numpifyCache ()
        grad = lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta))

        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)

    def appendCache (self, p, v, m, pb, gb, r1=None, r2=None) :
        """ Called every iteration of optimize() """

        if self.cache :
            self.pcache.append(np.copy(p))
            self.vcache.append(np.copy(v))
            self.mcache.append(np.copy(m))
            self.pbcache.append(np.copy(pb))
            self.gbcache.append(np.copy(gb))
            if r1 is not None : self.r1cache.append(np.copy(r1))
            if r2 is not None : self.r2cache.append(np.copy(r2))

    def numpifyCache (self) :
        """ Sets cache list in numpy format. Called before exit of optimize() """

        if self.cache :
            self.pcache = np.array(self.pcache)
            self.vcache = np.array(self.vcache)
            self.mcache = np.array(self.mcache)
            self.pbcache = np.array(self.pbcache)
            self.gbcache = np.array(self.gbcache)
            self.r1cache = np.array(self.r1cache)
            self.r2cache = np.array(self.r2cache)

    def replay (self, seed, c1=0.7, c2=0.7, alpha=1.2, beta=0.9) :
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
            vel = pso.vclip(vel, self.vmax)
            part, vel = pso.ipcd(part, vel, self.llim, self.rlim, alpha)

            less = self.obj(part) < self.obj(pb)
            pb[less] = part[less]
            gb = min(pb , key = lambda x : self.obj(x.reshape(1, -1))[0])

            pcache.append(part)
            vcache.append(vel)
            mcache.append(mom)
            pbcache.append(pb)
            gbcache.append(gb)

        return np.array(pcache), np.array(vcache), np.array(mcache), np.array(pbcache), np.array(gbcache)


class HECS_PSO (pso.PSO) :
    """
    Name    - A Hybrid Particle Swarm Algorithm with Embedded Chaotic Search
    Author  - Meng et. al.
    Link    - https://ieeexplore.ieee.org/document/1460442
    """

    def __init__ (self, obj, llim, rlim, Np, stag_tol=1e-3, Nc=6, Gmax=500, rrat=0.2, vrat=0.1) :
        """
        Constructor for the hybrid embedded chaotic search PSO optimizer -
            stag_tol        - Stagnation tolerance for kicking in chaotic search
            Nc              - Number of iterations to check for stagnation
            Gmax            - Maximum iterations in the chaotic search
            rrat            - Carrier wave radius in chaotic search

        Rest are defined in the base class
        """

        super().__init__(obj, llim, rlim, Np, vrat)
        self.stag_tol = stag_tol
        self.Nc = Nc
        self.Gmax = Gmax
        self.rrat = rrat
        self.cgen = None

    def __str__ (self) :
        """ Optimizer descriptor """
        return "Hybrid Embedded Chaotic Search PSO"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        fitness_q = deque(maxlen=self.Nc)
        fitness_q.append(self.obj(self.particles))
        return fitness_q, pbest, gbest

    def optimize (self, w=0.7, c1=1.7, c2=1.7, alpha=1.2,
                max_iters=10000, tol=1e-2,
                print_iters=False) :
        """ Runs the PSO loop """

        fitness_q, pbest, gbest = self._optim_init()

        # Set the chaotic generator if not previously set
        if self.cgen is None :
            self.cgen = cg.Logistic((self.Gmax, self.D), gens=1)

        i = -1
        while True :
            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

            # Chaotic search
            if i >= self.Nc :
                fits_ps = np.array(fitness_q).transpose()
                for j, fits_p in enumerate(fits_ps) :
                    if ((fits_p - self.obj(gbest.reshape(1, -1)))/fits_p < self.stag_tol).all() :
                        chaos_points = self.particles[j] + self.rrat*(self.rlim - self.llim)*(2*self.cgen.chaosPoints(1) - 1)
                        obj_cp = np.where(np.logical_and(self.llim.reshape(1, -1) <= chaos_points,
                                                        chaos_points <= self.rlim.reshape(1, -1)).all(axis=1),
                                        self.obj(chaos_points),
                                        np.inf)
                        gbest_p = np.argmin(obj_cp).flatten()[0]

                        # Update after chaotic search if feasible
                        if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(self.particles[j]) :
                            self.velocity[j] = self.particles[j] - chaos_points[gbest_p]
                            self.particles[j] = chaos_points[gbest_p]

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest, key = self.objkey)
            self.conv_curve.append(self.objkey(gbest))

            # Appends fitness for tracking whether to enter chaotic search
            fitness_q.append(self.obj(self.particles))

            # Velocity update
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

        grad = lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*w)
        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)


class PWLC_PSO (pso.PSO) :
    """
    Name    - An improved particle swarm optimization algorithm combined with piecewise linear chaotic map
    Author  - Xiang et. al.
    Link    - https://www.sciencedirect.com/science/article/abs/pii/S0096300307002081
    """

    def __init__ (self, obj, llim, rlim, Np, mu=0.7, rrat=0.8, rho=0.9, vrat=0.1) :
        """
        Constructor for the hybrid embedded chaotic search PSO optimizer -
            mu      - Parameter for the piecewise linear chaotic map
            rrat    - Ratio in terms of dimension size for the chaotic search radius
            rho     - Reduction factor for the search radius

        Rest are defined in the base class
        """

        super().__init__(obj, llim, rlim, Np, vrat)
        self.mu = mu
        self.rrat = rrat
        self.rho = rho
        self.cgen = None

    def __str__ (self) :
        """ Optimizer descriptor """
        return "Piece-wise Linear Chaotic PSO"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        return pbest, gbest

    def optimize (self, w=0.7, c1=1.7, c2=1.7, alpha=1.2,
                chaos_iters=500, max_pso_iters=10000, tol=1e-2,
                print_iters=False) :
        """ Optimization loop of plain PSO """

        pbest, gbest = self._optim_init()

        # Set the chaotic generator if not previously set
        if self.cgen is None :
            self.cgen = cg.Tent((chaos_iters, self.D), mu=self.mu, gens=1)

        i = 0
        while True :
            # Velocity update equation
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest_ind = np.argmin(self.obj(pbest)).flatten()[0]

            # Chaotic search
            cp = pbest[gbest_ind] + self.rrat*(self.rlim - self.llim)*(2*self.cgen.chaosPoints(1) - 1)
            obj_cp = np.where(np.logical_and(self.llim.reshape(1,-1) <= cp, cp <= self.rlim.reshape(1,-1)).all(axis=1),
                            self.obj(cp),
                            np.inf)
            gbest_p = np.argmin(obj_cp).flatten()[0]

            # Update after chaotic search if feasible
            if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(pbest[gbest_ind]) :
                new_vel = cp[gbest_p] - self.particles[gbest_ind]
                self.velocity[gbest_ind] = np.random.rand(self.D)*self.vmax*new_vel/np.linalg.norm(new_vel)
                pbest[gbest_ind] = self.particles[gbest_ind] = cp[gbest_p]

            # Copy gbest
            gbest = pbest[gbest_ind]
            self.conv_curve.append(self.objkey(gbest))
            self.rrat *= self.rho

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_pso_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        grad = lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*w)
        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)


class PWLC_EMPSO (pso.PSO) :
    """
    PWLCPSO with momentum
    """

    def __init__ (self, obj, llim, rlim, Np, mu=0.7, rrat=0.8, rho=0.9, vrat=0.1) :
        """
        Constructor for the hybrid embedded chaotic search PSO optimizer -
            mu      - Parameter for the piecewise linear chaotic map
            rrat    - Ratio in terms of dimension size for the chaotic search radius
            rho     - Reduction factor for the search radius

        Rest are defined in the base class
        """

        super().__init__(obj, llim, rlim, Np, vrat)
        self.mu = mu
        self.rrat = rrat
        self.rho = rho
        self.cgen = None

    def __str__ (self) :
        """ Optimizer descriptor """
        return "Piece-wise Linear Chaotic PSO"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        return np.zeros_like(pbest), pbest, gbest

    def optimize (self, beta=0.9, c1=0.7, c2=0.7, alpha=1.2,
                chaos_iters=500, max_pso_iters=10000, tol=1e-2,
                print_iters=False) :
        """ Optimization loop of plain PSO """

        momentum, pbest, gbest = self._optim_init()

        # Set the chaotic generator if not previously set
        if self.cgen is None :
            self.cgen = cg.Tent((chaos_iters, self.D), mu=self.mu, gens=1)

        i = 0
        while True :
            # Momentum update
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest_ind = np.argmin(self.obj(pbest)).flatten()[0]

            # Chaotic search
            cp = pbest[gbest_ind] + self.rrat*(self.rlim - self.llim)*(2*self.cgen.chaosPoints(1) - 1)
            obj_cp = np.where(np.logical_and(self.llim.reshape(1,-1) <= cp, cp <= self.rlim.reshape(1,-1)).all(axis=1),
                            self.obj(cp),
                            np.inf)
            gbest_p = np.argmin(obj_cp).flatten()[0]

            # Update after chaotic search if feasible
            if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(pbest[gbest_ind]) :
                new_vel = cp[gbest_p] - self.particles[gbest_ind]
                self.velocity[gbest_ind] = np.random.rand(self.D)*self.vmax*new_vel/np.linalg.norm(new_vel)
                momentum[gbest_ind] = 0
                pbest[gbest_ind] = self.particles[gbest_ind] = cp[gbest_p]

            # Copy gbest
            gbest = pbest[gbest_ind]
            self.conv_curve.append(self.objkey(gbest))
            self.rrat *= self.rho

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_pso_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        grad = lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta))
        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)
