import numpy as np

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
    Name    - Boundary Handling Approaches in Particle Swarm Optimization
    Author  - Padhye et. al.
    Link    - https://link.springer.com/chapter/10.1007/978-81-322-1038-2_25
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
    particleV = particles[vio]

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
        return gbest, np.sum((np.abs(self.particles - gbest) < tol).all(axis=1))
