import numpy as np

class CPSO_Optimizer () :
    """Class from the chaotic particle swarm optimizer
    Can also work as a vanilla PSO optimizer"""

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
            vel = CPSO_Optimizer.vclip(vel, self.vmax)
            part, vel = CPSO_Optimizer.ipcd(part, vel, self.llim, self.rlim, alpha)

            less = self.obj(part) < self.obj(pb)
            pb[less] = part[less]
            gb = min(pb , key = lambda x : self.obj(x.reshape(1,-1))[0])

            pcache.append(part)
            vcache.append(vel)
            mcache.append(mom)
            pbcache.append(pb)
            gbcache.append(gb)

        return np.array(pcache), np.array(vcache), np.array(mcache), np.array(pbcache), np.array(gbcache)

    def getSwarm (shape, llim, rlim, initgen="", randgen="", cache=False) :
        """
        Returns a swarm of particles which PSO uses
            shape   - (Np, D)
            llim    - Left limits for each dimension
            rlim    - Right limits for each dimension
            initgen - Number generator for initialising particles
                    - Can be basic numpy.random or chaotic generator
            randgen - Number generator for r1, r2 of PSO
                    - Can be basic numpy.random or chaotic generator
        """
        import chaosGen as cg

        # Dimensions of the swarm must be the no. of dimensions in the boundary
        assert shape[1] == len(llim) == len(rlim)

        # Chaotic map descriptor string --> Generator
        gs = lambda st : (lambda i : np.random.random_sample(shape)) if st == ""\
                                                                    else cg.ChaosGenerator.getGen(shape, st)

        # CPSO_Optimizer object ready to be initialised as it's supplied an objective function
        return lambda o : CPSO_Optimizer(o, llim, rlim,
                                             gs(initgen),
                                             gs(randgen),
                                             cache=cache)

    def resetCache (self) :
        """" Resets cache to empty. Called before optimize() """

        self.__setcache__()

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

    def __setcache__ (self) :
        """
        Sets all the caches to the empty list if cache parameter
        in initialisation is true
        """

        ######################################################################
        # Caches to hold optimization iterations for the last optimization performed
        # Contains  - position
        #           - velocity
        #           - momentum
        #           - pbest
        #           - gbest
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

    def __init__ (self, obj, llim, rlim, initgen, randgen, vrat=0.01, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            obj         - Objective function
            llim        - Left boundaries
            rlim        - Right boundaries
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator
            vrat        - Maximum velocity ratio dimensions used for clipping
        """

        self.obj = obj
        self.llim = llim
        self.rlim = rlim
        self.vrat = vrat
        self.initgen = initgen
        self.randgen = randgen

        # Defined as 'vrat' times the search space per dimension
        self.vmax = vrat*(rlim - llim).reshape(1,-1)
        self.cache = cache
        self.__setcache__()

    def initParticles (self) :
        """
        Initialises particle position and velocity.
        Called at beginning of optimize()
        """

        # The only way to find the dimension of the swarm
        D = len(self.llim)

        # Uses the first internal generator --> initgen(1)
        self.particles = np.array([l + (r-l)*self.initgen(1).transpose()[ind] \
                              for ind, l, r in zip(range(0,D), self.llim, self.rlim)]).transpose()

        # Uses the second internal generator --> initgen(2)
        self.velocity = np.array([self.vrat*(r-l)*(2*self.initgen(2).transpose()[ind] - 1)\
                              for ind, l, r in zip(range(0,D), self.llim, self.rlim)]).transpose()

    def vclip (velocity, vmax) :
        """ Clips and scales down the velocity according to 'vmax' """

        #####################################################################
        # for particle in swarm :
        #   ratio = 1
        #   for dim in dimensions :
        #       ratio = max (ratio, v[particle][dim]/vmax[dim])
        #   v[particle] /= ratio
        ######################################################################
        velocity /= (lambda x:np.where(x < 1, 1, x))\
                        (np.max(np.abs(velocity)/(vmax), axis=1, keepdims=True))

        return velocity

    def ipcd (particles, velocity, llim, rlim, alpha=1.2) :
        """
        Check 'Boundary Handling Approaches in Particle Swarm Optimization'
        by Padhye et. al.
        """

        part = particles + velocity

        leftvio = part < llim
        rightvio = part > rlim
        leftRight = np.logical_or(part < llim, part > rlim)
        vio = np.sum (leftRight, axis=1).astype(bool)
        viosum = np.sum(vio)

        if viosum == 0 :
            return part, velocity

        leftvio = leftvio[vio]
        rightvio = rightvio[vio]
        partV = part[vio]
        particleV = particle[vio]

        limvec = np.copy(partV)
        limvec[leftvio] = np.tile(llim, (viosum, 1))[leftvio]
        limvec[rightvio] = np.tile(rlim, (viosum, 1))[rightvio]

        diff = partV - particleV
        Xnot = np.sqrt (np.sum (np.square(diff), axis=1, keepdims=True))
        kvec = np.min (np.where (diff == 0, 1, (limvec - particleV)/diff), axis=1, keepdims=True)

        bvec = particleV + np.where (diff == 0, 0, kvec*diff)
        dvec = np.sqrt (np.sum (np.square(partV - bvec), axis=1, keepdims=True))

        Xpp = dvec*(1 + alpha*np.tan(\
                np.random.rand(viosum).reshape(-1,1)*np.arctan((Xnot - dvec)/(alpha * dvec))))
        boundk = (Xnot - Xpp)/Xnot

        part[vio] = particleV + np.where (diff == 0, 0, boundk*diff)
        velocity[leftRight] *= -1

        return part, velocity

    def optimize (self, c1=2, c2=2, alpha=1.2, beta=0.9, max_iters=10000, tol=1e-4) :
        """
        Performs the PSO optimization loop
        Arguments are default PSO parameters
        Returns the optimum found, and lambda function for approximate gradient
        """

        self.resetCache()
        self.initParticles()
        pbest = np.copy(self.particles)
        momentum = np.zeros(shape=pbest.shape)
        gbest = min(pbest, key = lambda x : self.obj(x.reshape(1,-1))[0])

        # Initial append to class cache
        self.appendCache(self.particles, self.velocity, momentum, pbest, gbest)

        ######################################################################
        # MAIN OPTIMIZATION LOOP
        ######################################################################
        i = 0
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1 = self.randgen(1)
            r2 = self.randgen(2)

            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = CPSO_Optimizer.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Check function docstring for details
            ######################################################################
            self.particles, self.velocity = CPSO_Optimizer.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = lambda x : self.obj(x.reshape(1,-1))[0])

            # Append to cache after updating particles, velocities, pbest and gbest
            self.appendCache (self.particles, self.velocity, momentum, pbest, gbest, r1, r2)

            i += 1
            print("\r{}".format(i), end="")
            if i == max_iters or \
            i > 100 and (self.velocity < tol).all() :
                break

        # Convert cache list to numpy ndarray
        self.numpifyCache ()

        return gbest, lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta))
