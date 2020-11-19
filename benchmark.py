import numpy as np
import pso
import cpso
import ripso

class Benchmark () :
    """
    Class that benchmarks a PSO variant. Implements the following metrics -
        1. Iterations
        2. Minima
        3. Mean fitness of all particles
        4. No. of particles converged
        5. Minima precision error
        6. Argmin precision error (euclidean norm)
        7. Success ratio
        8. Convergence curve (gbest vs. iters)
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for benchmarking object that takes in a PSO optimizer """

        self.pso_class = pso_class
        self.Np = Np
        self.spec = {
            'rspec' : None,
            'ospec' : None
        }

    def __str__ (self) :
        """ Name of benchmarking function """
        return "{} {}-D".format(self.obj_str, self.dims)

    @property
    def objkey (self) :
        """ Returns the objective function for single domain point """
        return lambda x : self.obj(x.reshape(-1, 1))[0]

    def eval (self, runs, succ_tol=1e-2, print_iters=True) :
        """ Evaluates the optimizer and computes benchmark properties """

        # Running specs
        rspec = {
            'iters'         : [],
            'minima'        : [],
            'mean_fitness'  : [],
            'no_conv'       : [],
            'min_err'       : [],
            'argmin_err'    : [],
            'iters'         : []
        }

        min_conv_curve, max_conv_curve = None, None
        min_iters, max_iters = np.inf, 0
        for _ in range(runs) :
            mizer = self.pso_class(self.obj, self.llim, self.rlim, self.Np)
            retpack = mizer.optimize(print_iters=print_iters)

            opt = retpack['rets'][0]
            iters = retpack['kwrets']['iters']
            opt_val = self.objkey(opt)

            rspec['iters'].append(iters)
            rspec['minima'].append(opt_val)
            rspec['mean_fitness'].append(
                np.mean(self.obj(retpack['kwrets']['particles']))
            )
            rspec['no_conv'].append(retpack['kwrets']['no_conv'])
            rspec['min_err'].append(opt_val - self.min)
            rspec['argmin_err'].append(np.min(
                np.linalg.norm(opt - self.argmins, axis=1)
            ))

            if iters < min_iters :
                min_conv_curve = retpack['kwrets']['conv_curve']
                min_iters = iters
            if iters > max_iters :
                max_conv_curve = retpack['kwrets']['conv_curve']
                max_iters = iters

        rspec = {
            s : np.array(arr)
            for s, arr in rspec.items()
        }

        ospec = {
            'mean_' + s : np.mean(arr)
            for s, arr in rspec.items()
        }

        ospec['conv_curves'] = (min_conv_curve, max_conv_curve)
        ospec['succ_ratio'] = np.sum(rspec['argmin_err'] < succ_tol)/iters
        self.spec['rspec'] = rspec
        self.spec['ospec'] = ospec

        return self.spec


class Sphere (Benchmark) :
    """
    Sphere function -
        sum(xi^2)
        f(0, ... ) = 0
        xi <- [...]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for sphere benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Sphere"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-np.inf, dims), np.repeat(np.inf, dims)
        self.argmins = np.repeat(0, dims).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Sphere function """
        return lambda X :np.sum (np.square(X), axis=1)

    @property
    def objder (self) :
        """ Derivative of Sphere function """
        return lambda x : 2*x


class Matyas (Benchmark) :
    """
    Matyas function -
        0.26*(x^2 + y^2) - 0.48xy
        f(0, 0) = 0
        x,y <- [-10, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for matyas benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Matyas"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-10, 2), np.repeat(10, 2)
        self.argmins = np.repeat(0, 2).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Matyas function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return 0.26*np.sum(np.square(xy), axis=1) - 0.48*x*y

        return f


class Schaffer4 (Benchmark) :
    """ Schaffer-4 function -
        0.5 + (cos^2(sin|x^2 - y^2|) - 0.5)/(1 + 0.001(x^2 + y^2))^2
        f(0, 1.25313) = 0.292579
        f(0, -1.25313) = 0.292579
        f(1.25313, 0) = 0.292579
        f(-1.25313, 0) = 0.292579
        x,y <- [-100, 100]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for schaffer4 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Schaffer4"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-100, 2), np.repeat(100, 2)
        self.argmins = np.array([
            [x, y]
            for x in [0, 1.25313, -1.25313]
            for y in [0, 1.25313, -1.25313]
            if x or y
        ])
        self.min = 0.292579

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x2 = np.square (xy[:,0])
            y2 = np.square (xy[:,1])
            return 0.5 + (np.square(np.cos(np.sin(np.abs(x2 - y2)))) - 0.5)/np.square(1 + 0.001*(x2 + y2))

        return f


class Rastrigin (Benchmark) :
    """
    Rastrigin function -
        An + sum(xi^2 - Acos(2pi*xi))
        f(0, ...) = 0
        xi <- [-5.12, 5.12]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for sphere benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Rastrigin"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-5.12, dims), np.repeat(5.12, dims)
        self.argmins = np.repeat(0, dims).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Rastrigin function """
        def f (X) :
            A = 10
            return A*X.shape[1] + np.sum(np.square(X) - A*np.cos(2*np.pi*X) , axis=1)

        return f

    @property
    def objder (self) :
        """ Derivative of Rastrigin function """
        return lambda x : 2*x + 20*np.pi*np.sin(2*np.pi*x)


benches = {
    'sphere'        : lambda p, dims    : Sphere(p, dims),
    'matyas'        : lambda p          : Matyas(p),
    'schaffer4'     : lambda p          : Schaffer4(p),
    'rastrigin'     : lambda p, dims    : Rastrigin(p, dims)
}

optimizers = {
    'vanilla'           : pso.PSO,
    'adaswarm'          : cpso.Adaswarm,
    'hecs'              : cpso.HECS_PSO,
    'pwlc'              : cpso.PWLC_PSO,
    'gb'                : cpso.GB_PSO,
    'ripso'             : ripso.RI_PSO
}
