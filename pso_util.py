import numpy as np
from itertools import cycle

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

def get_dirrand (point, nx, llim, rlim) :
    """
    Returns a random point from a line
        point       - Origin point of line
        nx          - Unit vector from line
        llim        - Left boundary of search space
        rlim        - Right boundary of search space
    """

    end = np.where (nx < 0, llim, rlim)
    diff = end - point
    min_k = np.argmin(np.abs(diff))
    seedlim = point + diff[min_k]/nx[min_k]*nx

    linD = np.array([np.linspace(a, b, 1000)[500:] for a, b in zip(point, seedlim)]).transpose()
    return linD[np.random.choice(np.arange(linD.shape[0]))]

def get_dirmin (point, nx, objkey, llim, rlim) :
    """
    Returns a minimum point from a line
        point       - Origin point of line
        nx          - Unit vector from line
        objkey      - Objective function by which to choose the minima
        llim        - Left boundary of search space
        rlim        - Right boundary of search space
    """

    end = np.where (nx < 0, llim, rlim)
    diff = end - point
    min_k = np.argmin(np.abs(diff))
    seedlim = point + diff[min_k]/nx[min_k]*nx

    linD = np.array([np.linspace(a, b, 1000)[500:] for a, b in zip(point, seedlim)]).transpose()
    lin_func = np.array([
        objkey(x) for x in linD
    ])
    min_lin = np.argmin(lin_func)
    return linD[min_lin]


def hull_hyperplanes (simplices, verts) :
    """ Get the hyperplane determinants and supports for the hull """

    # Determinants and supports for solving the 'point in polytope'
    dets, supps = [], []
    dims = verts.shape[1]
    for simp in simplices :
        ps = np.array([
            np.copy(verts[i]) for i in simp
        ])

        # Choose last point as the support
        supp = np.copy(ps[-1])
        ps = ps[:-1]
        ps -= supp

        # Evaluate determinants that will become the coefficients of the hyperplane equation
        det = np.array([
            sgn*np.linalg.det(ps[:, list(range(i)) + list(range(i+1, dims))])
            for i, sgn in zip(range(dims), cycle([1, -1]))
        ])

        dets.append(det)
        supps.append(supp)

    return dets, supps

def isPointIn (qp, pip, dets, supps) :
    """
    Checks if a point is inside a convex hull
        qp          - query point
        hull_tup    - convex hull tuple with associated information
    """

    #####################################################################
    # Sign of hyperplane equation, when both 'qp' and 'pip' are substituted
    # into it,
    #####################################################################
    return np.all([
        np.sum(d*(pip-s)) * np.sum(d*(qp-s)) >= 0
        for d, s in zip(dets, supps)
    ])
