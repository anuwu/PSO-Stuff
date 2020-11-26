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
        self.vmax = vrat*(rlim - llim).reshape(1,-1)              # Maximum velocity for velocity clipping
        self.conv_curve = []

    def __str__ (self) :
        """ Optimizer descriptor """
        return "Vanilla PSO"

    def _initp (self, Nparts, Nvels=None) :
        parts = np.array([l + (r-l)*np.random.rand(self.D, Nparts)[ind] \
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

        if Nvels is None : Nvels = Nparts
        vels = np.array([self.vrat*(r-l)*(2*np.random.rand(self.D, Nvels)[ind] - 1)\
                              for ind, l, r in zip(range(0, self.D), self.llim, self.rlim)]).transpose()

        return parts, vels

    def initParticles (self, rad_init=None) :
        """ Initialises particle position and velocity """

        if rad_init is None :
            self.particles, self.velocity = self._initp(self.Np)
        else :
            rad_cent, rad_ps, min_rad = rad_init
            rad_points = rad_cent + min_rad + 2*self.vmax*(2*np.random.rand(rad_ps, self.D) - 1)
            rad_points = rad_points[
                np.logical_and(
                self.llim.reshape(1, -1) <= rad_points,
                rad_points <= self.rlim.reshape(1, -1)
                ).all(axis=1)
            ]
            legit_rad = len(rad_points)

            rand_points, self.velocities = self._initp(self.Np - legit_rad, self.Np)
            self.particles = np.concatenate([rad_points, rand_points], axis=0)

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        # Initialises swarm position and velocity
        self.initParticles()
        pbest = np.copy(self.particles)
        gbest = np.copy(min(pbest, key = self.objkey))

        self.conv_curve = [self.objkey(gbest)]
        return pbest, gbest

    def optimize (self, w=0.7, c1=1.7, c2=1.7, alpha=1.2,
                max_iters=10000, tol=1e-2,
                print_iters=False) :
        """ Optimization loop of plain PSO """

        pbest, gbest = self._optim_init()

        i = 0
        while True :
            # Velocity update
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            self.velocity = w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # pbest, gbest update
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key=self.objkey)
            self.conv_curve.append(self.objkey(gbest))

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        grad = lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*w)
        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)

    def optRet (self, gbest, grad, tol, iters) :
        """
        Returns a tuple of necessary information
        after completition of optimization loop
        """

        return {
            'rets'      : (gbest, grad),
            'kwrets'    : {
                'iters'         : iters,
                'no_conv'       : np.sum((np.abs(self.particles - gbest) < tol).all(axis=1)),
                'particles'     : np.copy(self.particles),
                'conv_curve'    : np.array(self.conv_curve)
            }
        }
