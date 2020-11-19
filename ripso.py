import numpy as np
import pso
from itertools import cycle
from scipy.spatial import ConvexHull

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

class Hull () :
    """
    Contains information of the minima hulls
    used by RI_PSO and associated functions
    """

    def __hyperplanes__ (self) :
        """ Sets the hyperplane equation of the hull faces """

        dets, supps = [], []
        dims = self.verts.shape[1]
        for simp in self.hull.simplices :
            ps = np.array([
                np.copy(self.verts[i]) for i in simp
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

        self.dets = dets
        self.supps = supps

    def __init__ (self, verts) :
        """ Constructor of the Hull object """

        self.verts = verts
        self.hull = ConvexHull(verts)
        self.pip = (verts[0] + verts[(len(verts) - 1)//2])/2
        self.max_vert_dist = np.max(np.linalg.norm(self.verts - self.pip, axis=1))

        self.__hyperplanes__ ()

    def isPointIn (self, qp) :
        """
        Checks if a point is inside a convex hull
            qp          - query point
            hull_tup    - convex hull tuple with associated information
        """

        ## No need to check further if query point is further than the largest
        ## distance of a hull vertices from the pip (point in polytope)
        if np.linalg.norm(qp - self.pip) > self.max_vert_dist :
            return False

        ## Sign of hyperplane equation, when both 'qp' and 'pip' are substituted
        ## into it,
        return np.all([
            np.sum(d*(self.pip-s)) * np.sum(d*(qp-s)) >= 0
            for d, s in zip(self.dets, self.supps)
        ])


class RI_PSO (pso.PSO) :
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

    def __str__ (self) :
        """ Optimizer descriptor """
        return "Reverse-Informed PSO"

    def optimize (self, runs=5, cent_init_rat=0.5, trap_rat=0.20, print_iters=False) :
        """ Optimization loop involving forward() and reverse() """

        if print_iters : print("Run 1")
        ret = self.forward(print_iters=print_iters)
        opt = ret['rets'][0]
        self.reverse(opt, print_iters=print_iters)

        min_opt, min_optval = opt, self.objkey(opt)
        min_ret, min_hull = ret, self.hulls[-1]
        for i in range(runs-1) :
            if print_iters : print(f"Run {i+2}")
            ret = self.forward((min_opt,
                                np.ceil(cent_init_rat*self.Np).astype(np.uint),
                                min_hull.max_vert_dist
                                ),
                print_iters=print_iters
            )

            opt = ret['rets'][0]
            if opt is None or i == runs - 2 :
                break

            self.reverse(opt, print_iters=print_iters)
            optval = self.objkey(opt)
            if optval < min_optval :
                min_opt, min_optval = opt, optval
                min_ret, min_hulls = ret, self.hulls[-1]

        if print_iters : print("\n", end="")
        return min_ret

    def forward (self, rad_init=None, c1=2, c2=2, alpha=1.2, beta=0.9,
                max_iters=10000, rad_search_points=500, tol=1e-2, trap_rat=0.20,
                print_iters=False) :
        """ Forward PSO with hull exclusion and radial search """

        # Initialise particles and necessary states

        self.initParticles(rad_init)
        pbest = np.copy(self.particles)
        momentum = np.zeros_like(self.particles)
        gbest = min(pbest, key=self.objkey)
        self.conv_curve = [self.objkey(gbest)]
        trap = []

        i = 0
        while True :
            # Momentum and velocity update
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

            trap.append(0)
            # Hull exclusion
            if self.hulls != [] :
                for j, qp in enumerate(self.particles) :
                    for hull in self.hulls :
                        if hull.isPointIn(qp) :
                            # Central about about which radial search occurs within a search radius
                            rad_cent = pbest[np.argmin(np.array([
                                np.inf if k == j else self.objkey(pb)
                                for k, pb in enumerate(pbest)
                            ])).flatten()[0]]

                            rad_points = rad_cent + self.vmax*(2*np.random.rand(rad_search_points, self.D) - 1)

                            # Checking if radial particle violates dimension limits
                            lim_rp = np.logical_or(self.llim.reshape(1, -1) > rad_points, rad_points < self.rlim.reshape(1, -1)).any(axis=1)

                            # Checking if radial particle itself is in some minima hull
                            rp_out_hull = np.array([
                                np.array([
                                    h.isPointIn(rp)
                                    for h in self.hulls
                                ]).any()
                                for rp in rad_points
                            ])

                            # Disallow particle if it's violated search limits or within some hull
                            obj_rp = np.where(np.logical_or(lim_rp, rp_out_hull), np.inf, self.obj(rad_points))

                            # Best radial particle
                            gbest_rp = np.argmin(obj_rp).flatten()

                            # Replace original particle with best radial particle
                            self.particles[j] = rad_points[gbest_rp]
                            pbest[j] = self.particles[j]
                            self.velocity[j] = rad_cent - self.particles[j]
                            self.velocity[j] = self.vmax*self.velocity[j]/np.linalg.norm(self.velocity[j])
                            momentum[j] = 0

                            trap[-1] += 1
                            break

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest, key=self.objkey)
            self.conv_curve.append(self.objkey(gbest))

            i += 1
            if print_iters : print("\rForward = {}".format(i), end="")
            if i >= 25 and sum(trap[-25:])/25 >= np.ceil(trap_rat*self.Np) :
                print("\n", end="")
                return {
                    'rets'      : tuple(4*[None]),
                    'kwrets'    : {}
                }

            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        grad = lambda x : -(c1*np.sum(r1) + c2*np.sum(r2))*(x - gbest)/(len(r1)*(1-beta))
        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)

    def reverse (self, opt, w=0.7, c1=2, c2=2,
                min_iters=50, max_iters=1000,
                print_iters=False) :
        """ Reverse PSO loop """

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
                        get_dirmin(opt, nx, self.objkey, self.llim, self.rlim)
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
            vs = pso.vclip(vs, vmax)

            # Apply IPCD boundary handling, note the minus sign on 'vs'
            xs, vs = pso.ipcd(xs, -vs, self.llim, self.rlim)

            # Update pbest and gbest
            less = self.obj(xs) <= self.obj(pbest)
            xs[less] = np.copy(pbest[less])
            more = np.invert(less)
            pbest[more] = np.copy(xs[more])
            gbest = min(pbest, key=self.objkey)

            if print_iters : print("\rReverse = {}".format(i), end="")
            if i >= min_iters and less.all() :
                if not less_once :
                    less_once = True
                else :
                    break

        verts = np.copy(np.concatenate((xs, gbest.reshape(1, -1)), axis=0))
        self.hulls.append(Hull(verts))
        if print_iters : print("\n", end="")
