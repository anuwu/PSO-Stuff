#################################################################
# Should probably structure the variables among classes and
# sub-classes efficiently later. Also, I haven't checked whether
# the chaotic search part in PWLC_PSO and HECS_PSO work.
#################################################################

import numpy as np
import pso_util as pu
import chaosGen as cg
from collections import deque
from scipy.spatial import ConvexHull

class PSO () :
    """ Base class for a variety of PSO optimizers """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.1) :
        """
        Constructor of base PSO optimizer -
            obj         - Objective function to minimize
            llim        - Left limits in each dimension
            rlim        - Right limits in each dimension
            Np          - Number of particles in the swarm
            vrat        - Velocity limit ratio
        """

        self.obj = obj
        self.objkey = lambda x : self.obj(x.reshape(1, -1))[0]      # Evaluating objective function for a single point
        self.llim = llim
        self.rlim = rlim
        self.Np = Np
        self.D = len(llim)                                          # Only way to obtain the dimensions of the swarm
        self.vrat = vrat
        self.vmax = vrat*(rlim - llim).reshape(1,-1)                # Maximum velocity for velocity clipping

    def initParticles (self) :
        """ Initialises particle position and velocity """

        self.particles = np.array([l + (r-l)*np.random.rand(self.D, self.Np)[ind] \
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

        self.velocity = np.array([self.vrat*(r-l)*(2*np.random.rand(self.D, self.Np)[ind] - 1)\
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

    def __optiminit__ (self) :
        """ Initialiser of certain state variables before the optimization loop """

        # Initialises swarm position and velocity
        self.initParticles()
        pbest = np.copy(self.particles)
        gbest = np.copy(min(pbest, key = self.objkey))
        return pbest, gbest

    def optimize (self, w=0.7, c1=2, c2=2, min_iters=100, max_iters=10000, tol=1e-2) :
        """ Optimization loop of plain PSO """

        pbest, gbest = self.__optiminit__()

        i = 0
        while True :
            # Velocity update
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # pbest, gbest update
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key=self.objkey)

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > min_iters and (np.abs(self.particles - gbest) < tol).all() :
                break

        print("\n", end="")
        return gbest, (np.abs(self.particles - gbest) < tol).all()


class ChaoticAdaswarm (PSO) :
    """
    AdaSwarm with random number generators replaced with chaotic generators

    Name    - AdaSwarm: A Novel PSO optimization Method for the Mathematical Equivalence of Error Gradients
    Author  - Rohan et. al.
    Link    - https://arxiv.org/abs/2006.09875
    """

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

    def __init__ (self, obj, llim, rlim, Np, initgen, randgen, vrat=0.1, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator

        The rest are defined in the base class PSO()
        """

        super().__init__(obj, llim, rlim, Np, vrat)
        self.initgen = initgen                                  # Chaotic generator for the swarm initialiser
        self.randgen = randgen                                  # Chaotic generator for r1, r2 in optimization loop
        self.cache = cache                                      # PSO progress cache for replaying history (used in calculating LE of trajectory)
        self.__setcache__()

    def __optiminit__ (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super.__optiminit__()
        momentum = np.zeros_like(self.particles)
        self.appendCache(self.particles, self.velocity, momentum, pbest, gbest)
        return momentum, pbest, gbest

    def optimize (self, get_grad=False, c1=2, c2=2, alpha=1.2, beta=0.9, min_iters=100, max_iters=10000, tol=1e-2) :
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

            # Momentum and velocity update
            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = self.objkey)

            # Append to cache after updating particles, velocities, pbest and gbest
            self.appendCache (self.particles, self.velocity, momentum, pbest, gbest, r1, r2)

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > min_iters and (np.abs(self.particles - gbest) < tol).all() :
                break

        # Convert cache list to numpy ndarray
        self.numpifyCache ()

        print("\n", end="")
        return (gbest, (np.abs(self.particles - gbest) < tol).all()) + \
            (lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta)), ) if get_grad else ()

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
        self.cgen = None                                # Chaotic generator for chaotic search

    def __optiminit__ (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super().__optiminit__()
        fitness_q = deque(maxlen=self.Nc)
        fitness_q.append(self.obj(self.particles))
        return fitness_q, pbest, gbest

    def optimize (self, w=0.7, c1=2, c2=2, alpha=1.2, min_iters=100, max_iters=10000, tol=1e-2) :
        """ Runs the PSO loop """

        fitness_q, pbest, gbest = self.__optiminit__()

        # Set the chaotic generator if not previously set
        if self.cgen is None :
            self.cgen = cg.Logistic((self.Gmax, self.D), gens=1)

        i = -1
        while True :
            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > min_iters and (np.abs(self.particles - gbest) < tol).all() :
                break

            # Chaotic search
            if i >= self.Nc :
                fits_ps = np.array(fitness_q).transpose()
                for j, fits_p in enumerate(fits_ps) :
                    if ((fits_p - self.obj(gbest.reshape(1, -1)))/fits_p < self.stag_tol).all() :
                        chaos_points = self.particles[j] + self.rrat*(self.rlim - self.llim)*(2*self.cgen.chaosPoints(1) - 1)
                        obj_cp = np.where(np.logical_and(self.llim.reshape(1, -1) <= chaos_points,
                                                        chaos_points <= self.rlim.reshape(1, -1)),
                                        self.obj(chaos_points),
                                        np.inf)
                        gbest_p = np.argmin(obj_cp).flatten()

                        # Update after chaotic search if feasible
                        if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(self.particles[j]) :
                            self.velocity[j] = self.particles[j] - chaos_points[gbest_p]
                            self.particles[j] = chaos_points[gbest_chaos_points]

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest, key = self.objkey)

            # Appends fitness for tracking whether to enter chaotic search
            fitness_q.append(self.obj(self.particles))

            # Velocity update
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

        return gbest, (np.abs(self.particles - gbest) < tol).all()


class PWLC_PSO (PSO) :
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

    def __optiminit__ (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super().__optiminit__()
        return pbest, gbest

    def optimize (self, w=0.7, c1=2, c2=2, alpha=1.2, max_chaos_iters=500, min_pso_iters=100, max_pso_iters=10000, tol=1e-2) :
        """ Optimization loop of plain PSO """

        pbest, gbest = self.__optiminit__()

        # Set the chaotic generator if not previously set
        if self.cgen is None :
            self.cgen = cg.Tent((max_chaos_iters, self.D), mu=self.mu, gens=1)

        i = 0
        while True :
            # Velocity update equation
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest_ind = np.argmin(self.obj(pbest)).flatten()
            gbest = pbest[gbest_ind]

            # Chaotic search
            cp = gbest + self.rrat*(self.rlim - self.llim)*(2*self.cgen.chaosPoints(1) - 1)
            obj_cp = np.where(np.logical_and(self.llim.reshape(1,-1) <= cp, cp <= self.rlim.reshape(1,-1)), self.obj(cp), np.inf)
            gbest_p = np.argmin(obj_cp).flatten()

            # Update after chaotic search if feasible
            if obj_cp[gbest_p] != np.inf and obj_cp[gbest_p] < self.objkey(pbest[gbest_ind]) :
                self.velocity[gbest_ind] = 0
                pbest[gbest_ind] = cp[gbest_p]
                self.particles[gbest_ind] = pbest[gbest_ind]

            self.rrat *= self.rho
            i += 1
            print("\r{}".format(i), end="")
            if i == max_pso_iters or \
            i > min_pso_iters and (np.abs(self.particles - self.particles[0]) < tol).all() :
                break

        print("\n", end="")
        return gbest, (np.abs(self.particles - gbest) < tol).all()


class GB_PSO (PSO) :
    """
    Name        - An Adaptive Velocity Particle Swarm Optimization for High-Dimensional Function Optimization
    Author      - Arasomwan et. al.
    Link        - https://ieeexplore.ieee.org/document/6557850
    """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.1) :
        """ Constructor of GB_PSO, no additional parameters beyond plain PSO """

        super().__init__(obj, llim, rlim, Np, vrat)

    def __optiminit__ (self) :
        self.initParticles()
        gbest = np.copy(min(self.particles, key=self.objkey))
        return gbest

    def optimize (self, w=0.7, c1=2, c2=2, alpha=1.2, min_iters=100, max_iters=10000, tol=1e-2) :
        """ Optimization loop of plain PSO """

        gbest = self.__optiminit__()

        i = 0
        while True :
            # Velocity update
            max_dist = np.linalg.norm(gbest)
            p_dists = np.linalg.norm(self.particles - gbest, axis=1)
            self.velocity = w*self.velocity + (gbest - self.particles)*p_dists/max_dist

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > min_iters and (np.abs(self.particles - gbest) < tol).all() :
                break

        print("\n", end="")
        return gbest, (np.abs(self.particles - gbest) < tol).all()


class RI_PSO (PSO) :
    """ Reverse-Informed Particle Swarm Optimizer """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.1) :
        """
        Constructor of base PSO optimizer -
            obj         - Objective function to minimize
            llim        - Left limits in each dimension
            rlim        - Right limits in each dimension
            vrat        - Velocity limit ratio
        """
        super().__init__(obj, llim, rlim, Np, vrat)
        self.hulls = []
        self.cgen = None

    def forward (self, c1=2, c2=2, alpha=1.2, beta=0.9, min_iters=100, max_iters=10000, max_chaos_iters=500, tol=1e-2, chaos_search_prob=0.75) :
        """ Forward PSO with hull exclusion and chaotic search """

        # Initialise particles and necessary states
        self.initParticles()
        pbest = np.copy(self.particles)
        momentum = np.zeros_like(self.particles)
        gbest_ind = np.argmin(self.obj(pbest)).flatten()[0]

        # Set the chaotic generator if not previously set
        if self.hulls != [] and self.cgen is None :
            self.cgen = cg.Tent((max_chaos_iters, self.D), mu=0.49999, gens=1)

        i = 0
        while True :
            # Momentum and velocity update
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(pbest[gbest_ind] - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pu.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pu.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Hull exclusion
            if self.hulls != [] :
                for j, qp in enumerate(self.particles) :
                    for hull_tup in self.hulls :
                        pip = hull_tup[1]
                        dets, supps = hull_tup[3:]
                        if pu.isPointIn(qp, pip, dets, supps) :
                            if np.random.rand() > 1 - chaos_search_prob :
                                # Copying pbest
                                pbest_tmp = np.array([
                                    np.inf if k == j else self.objkey(pb)
                                    for k, pb in enumerate(pbest)
                                ])

                                # Central about about which chaotic search occurs within a search radius
                                chaos_cent = pbest[np.argmin(pbest_tmp)]
                                chaos_points = chaos_cent + self.vmax*(2*self.cgen.chaosPoints(1) - 1)
                                # chaos_points = chaos_cent + self.vmax*(2*np.random.rand(max_chaos_iters, self.D) - 1)

                                # Checking if chaotic particle violates dimension limits
                                lim_cp = np.logical_or(self.llim.reshape(1, -1) > chaos_points, chaos_points < self.rlim.reshape(1, -1)).any(axis=1)

                                # Checking if chaotic particle itself is in some minima hull
                                cp_out_hull = np.array([
                                    np.array([
                                        pu.isPointIn(cp, ht[1], ht[3], ht[4])
                                        for ht in self.hulls
                                    ]).any()
                                    for cp in chaos_points
                                ])

                                # Disallow particle if it's violated search limits or within some hull
                                obj_cp = np.where(np.logical_or(lim_cp, cp_out_hull), np.inf, self.obj(chaos_points))

                                # Best chaotic particle
                                gbest_cp = np.argmin(obj_cp).flatten()

                                # Replace original particle with best chaotic particle
                                self.particles[j] = chaos_points[gbest_cp]
                            else :
                                while True :
                                    randp = self.llim + np.random.rand(self.D)*(self.rlim - self.llim)
                                    in_hull = np.array([
                                        pu.isPointIn(randp, ht[1], ht[3], ht[4])
                                        for ht in self.hulls
                                    ]).any()

                                    if not in_hull :
                                        break

                                self.particles[j] = randp

                            pbest[j] = self.particles[j]
                            self.velocity[j] = self.vmax*(2*np.random.rand(self.D) - 1)
                            momentum[j] = 0
                            break

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest_ind = np.argmin(self.obj(pbest)).flatten()[0]

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > min_iters and (np.abs(self.particles - pbest[gbest_ind]) < tol).all() :
                break

        # print("\rForward iterations = {}\n".format(i), end="")
        print("\n", end="")
        return pbest[gbest_ind], (np.abs(self.particles - pbest[gbest_ind]) < tol).all()

    def reverse (self, opt, w=0.7, c1=2, c2=2, min_iters=50, max_iters=1000) :
        """ Reverse PSO loop """

        # xs, fs = pu.dirmin_foreign(opt, self.objkey, self.llim, self.rlim, self.Np)
        xs = opt + 1e-3*(np.random.rand(self.Np, self.D) - 0.5)
        vs = 1e-3*np.random.rand(self.Np, self.D)
        pbest = np.copy(xs)
        gbest = min(pbest, key=self.objkey)

        # Same magintude as forward PSO stopping criteria tolerance
        vmax = 1e-2*np.ones_like(self.llim).reshape(1, -1)
        less_once, fs = False, None

        for i in range(max_iters) :
            r1s = np.random.rand(self.Np, self.D)
            r2s = np.random.rand(self.Np, self.D)

            if less_once :
                if fs is None :
                    delta_xs = xs - opt
                    nxs = delta_xs/np.linalg.norm(delta_xs, axis=1, keepdims=True)
                    fs = np.array([
                        pu.get_dirmin(opt, nx, self.objkey, self.llim, self.rlim)
                        for nx in nxs
                    ])
                pb_past = fs
                gb = gbest
            else :
                pb_past = xs
                gb = opt

            # Each dimension of a particle has an associated update matrix
            mats = np.array([
                [
                    [1 - c1*r1s[p,d] - c2*r2s[p,d], w, c1*r1s[p,d]*pb_past[p,d] + c2*r2s[p,d]*gbest[d]],
                    [-c1*r1s[p,d] - c2*r2s[p,d], w, c1*r1s[p,d]*pb_past[p,d] + c2*r2s[p,d]*gbest[d]],
                    [0, 0, 1]
                ]
                for p in range(self.Np)
                for d in range(self.D)
            ]).reshape(self.Np, self.D, 3, 3)

            # Invert update matrices and find reverse position
            vecs = np.array([
                np.dot(np.linalg.inv(mats[p,d]), np.array([xs[p,d], vs[p,d], 1]))
                for p in range(self.Np)
                for d in range(self.D)
            ]).reshape(self.Np, self.D, 3)
            vs = vecs[...,1]

            # Velocity clipping applies
            vs = pu.vclip(vs, vmax)

            # Apply IPCD boundary handling, note the minus sign on 'vs'
            xs, vs = pu.ipcd(xs, -vs, self.llim, self.rlim)

            # Update pbest and gbest
            less = self.obj(xs) <= self.obj(pbest)
            xs[less] = np.copy(pbest[less])
            more = np.invert(less)
            pbest[more] = np.copy(xs[more])
            gbest = min(pbest, key=self.objkey)

            print("\r{}".format(i), end="")
            if i >= min_iters and less.all() :
                if not less_once :
                    less_once = True
                else :
                    break

        verts = np.copy(np.concatenate((xs, gbest.reshape(1, -1)), axis=0))
        hull = ConvexHull(verts)
        pip = (verts[0] + verts[(len(hull.vertices) - 1)//2])/2

        # print("\rReverse iterations = {}\n".format(i), end="")
        print("\n", end="")
        self.hulls.append((hull, pip, verts) + pu.hull_hyperplanes(hull.simplices, verts))
