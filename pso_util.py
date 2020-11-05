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
