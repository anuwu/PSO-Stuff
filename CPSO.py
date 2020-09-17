import numpy as np
import chaosGen as cg

class PSO_Optimizer () :
######################################################################
# Class from the chaotic particle swarm optimizer
# Can also work as a vanilla PSO optimizer
######################################################################

    ######################################################################
    # Caches to hold optimization iterations for the last optimization performed
    # Contains  - position
    #           - velocity
    #           - momentum
    #           - pbest
    #           - gbest
    ######################################################################
    pcache = []
    vcache = []
    mcache = []
    pbcache = []
    gbcache = []

    def getSwarm (shape, llim, rlim, initgen="", randgen="") :
    ######################################################################
    # Returns a swarm with defined shape and search boundary
    #
    # > shape   - (Np, D)
    # > llim    - Left limits for each dimension
    # > rlim    - Right limits for each dimension
    # > initgen - Number generator for initialising particles
    #           - Can be basic numpy.random or chaotic generator
    # > randgen - Number generator for r1, r2 of PSO
    # >         - Can be basic numpy.random or chaotic generator
    ######################################################################

        # Dimensions of the swarm must be the no. of dimensions in the boundary
        assert shape[1] == len(llim) == len(rlim)

        # Swarm string --> Generator
        gs = lambda st : (lambda i : np.random.random_sample (shape)) if st == ""\
                                                                    else cg.ChaosGenerator.getGen (shape, st)

        # PSO_Optimizer object ready to be initialised as it's supplied an objective function
        return lambda o : PSO_Optimizer (o, llim, rlim,
                                             gs(initgen),
                                             gs(randgen))

    def resetCache () :
    # Resets cache to empty. Called before optimize()

        PSO_Optimizer.pcache = []
        PSO_Optimizer.vcache = []
        PSO_Optimizer.mcache = []
        PSO_Optimizer.pbcache = []
        PSO_Optimizer.gbcache = []

    def appendCache (p, v, m, pb, gb) :
    # Called every iteration of optimize()

        PSO_Optimizer.pcache.append (np.copy(p))
        PSO_Optimizer.vcache.append (np.copy(v))
        PSO_Optimizer.mcache.append (np.copy(m))
        PSO_Optimizer.pbcache.append (np.copy(pb))
        PSO_Optimizer.gbcache.append (np.copy(gb))

    def numpifyCache () :
    # Sets cache list in numpy format. Called before exit of optimize()

        PSO_Optimizer.pcache = np.array (PSO_Optimizer.pcache)
        PSO_Optimizer.vcache = np.array (PSO_Optimizer.vcache)
        PSO_Optimizer.mcache = np.array (PSO_Optimizer.mcache)
        PSO_Optimizer.pbcache = np.array (PSO_Optimizer.pbcache)
        PSO_Optimizer.gbcache = np.array (PSO_Optimizer.gbcache)


    def __init__ (self, obj, llim, rlim, initer, rander, vrat=0.1) :
    ######################################################################
    # > obj         - Objective function
    # > llim        - Left boundaries
    # > rlim        - Right boundaries
    # > initer      - Position and velocity initialiser
    # > rander      - r1, r2 generator
    # > vrat        - Maximum velocity ratio dimensions used for clipping
    ######################################################################

        self.obj = obj
        self.llim = llim
        self.rlim = rlim
        self.vrat = vrat
        self.initgen = initer
        self.randgen = rander

        # Defined as 'vrat' times the search space per dimension
        self.vmax = vrat*(rlim - llim).reshape(1,-1)

    def initParticles (self) :
    # Initialises particle position and velocity. Called at beginning of optimize()

        # The only way to find the dimension of the swarm
        D = len(self.llim)

        # Uses the first internal generator --> initgen(1)
        self.particles = np.array([l + (r-l)*self.initgen(1).transpose()[ind] \
                              for ind, l, r in zip(range(0,D), self.llim, self.rlim)]).transpose()

        # Uses the second internal generator --> initgen(2)
        self.velocity = np.array([self.vrat*(r-l)*(2*self.initgen(2).transpose()[ind] - 1)\
                              for ind, l, r in zip(range(0,D), self.llim, self.rlim)]).transpose()


    def vclip (self) :
    # clips the velocity according to 'vmax'

        ######################################################################
        # for particle in swarm :
        #   ratio = 1
        #   for dim in dimensions :
        #       ratio = max (ratio, v[particle][dim]/vmax[dim])
        #   v[particle] /= ratio
        ######################################################################
        self.velocity /= (lambda x:np.where(x < 1, 1, x))\
                        (np.max(np.abs(self.velocity)/(self.vmax), axis=1, keepdims=True))


    def ipcd (self, alpha=1.2) :
    ######################################################################
    # Check 'Boundary Handling Approaches in Particle Swarm Optimization'
    # by Padhye et. al.
    # Mutates object directly
    ######################################################################

        part = self.particles + self.velocity

        leftvio = part < self.llim
        rightvio = part > self.rlim
        leftRight = np.logical_or(part < self.llim, part > self.rlim)
        vio = np.sum (leftRight, axis=1).astype(bool)
        viosum = np.sum(vio)

        if viosum == 0 :
            self.particles = part
            return

        leftvio = leftvio[vio]
        rightvio = rightvio[vio]
        partV = part[vio]
        particleV = self.particle[vio]

        limvec = np.copy(partV)
        limvec[leftvio] = np.tile (self.llim, (viosum, 1))[leftvio]
        limvec[rightvio] = np.tile (self.rlim, (viosum, 1))[rightvio]

        diff = partV - particleV
        Xnot = np.sqrt (np.sum (np.square(diff), axis=1, keepdims=True))
        kvec = np.min (np.where (diff == 0, 1, (limvec - particleV)/diff), axis=1, keepdims=True)

        bvec = particleV + np.where (diff == 0, 0, kvec*diff)
        dvec = np.sqrt (np.sum (np.square(partV - bvec), axis=1, keepdims=True))

        Xpp = dvec*(1 + alpha*np.tan(\
                np.random.rand(viosum).reshape(-1,1)*np.arctan((Xnot - dvec)/(alpha * dvec))))
        boundk = (Xnot - Xpp)/Xnot

        part[vio] = particleV + np.where (diff == 0, 0, boundk*diff)

        self.particles = part
        self.velocity[leftRight] *= -1

    def optimize (self, c1=2, c2=2, alpha=1.2, beta=0.9, iters=1000) :
    ######################################################################
    # Performs the PSO optimization loop
    # Arguments are default PSO parameters
    # Returns the optimum found, and lambda function for approximate gradient
    ######################################################################

        # Reset class cache
        PSO_Optimizer.resetCache()
        self.initParticles ()
        pbest = self.particles
        momentum = np.zeros (shape=pbest.shape)
        gbest = min (pbest , key = lambda x : self.obj(x.reshape(1,-1))[0])

        # Initial append to class cache
        PSO_Optimizer.appendCache (self.particles, self.velocity, momentum, pbest, gbest)

        ######################################################################
        # MAIN OPTIMIZATION LOOP
        ######################################################################
        for i in range (0, iters) :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1 = self.randgen(1)
            r2 = self.randgen(2)

            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.vclip()

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary Handling
            # Check function definition for details
            ######################################################################
            self.ipcd()

            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = self.particles[less]
            gbest = min (pbest , key = lambda x : self.obj(x.reshape(1,-1))[0])

            # Append to cache after updating particles, velocities, pbest and gbest
            PSO_Optimizer.appendCache (self.particles, self.velocity, momentum, pbest, gbest)

        # Convert cache list to numpy ndarray
        PSO_Optimizer.numpifyCache ()

        return gbest, lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta))
