import numpy as np

class Bench () :
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
        return lambda x : self.obj(x.reshape(1, -1))[0]

    def eval (self, runs, succ_tol=1e-2, print_iters=False) :
        """ Evaluates the optimizer and computes benchmark properties """

        # Running specs
        rspec = {
            'iters'         : [],
            'minima'        : [],
            'mean_fitness'  : [],
            'no_conv'       : [],
            'min_err'       : [],
            'argmin_err'    : [],
        }

        min_conv_curve, max_conv_curve = None, None
        min_iters, max_iters = np.inf, 0

        for i in range(runs) :
            if not print_iters : print(f"Run {i+1}")

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

        ospec['std_iters'] = np.std(rspec['iters'], ddof=1)
        ospec['std_minima'] = np.std(rspec['minima'], ddof=1)
        ospec['conv_curves'] = (min_conv_curve, max_conv_curve)
        ospec['succ_ratio'] = np.sum(rspec['argmin_err'] < succ_tol)/runs
        self.spec['rspec'] = rspec
        self.spec['ospec'] = ospec

        return self.spec


class Sphere (Bench) :
    """
    Sphere -
        sum(xi^2)
        f(0, ... ) = 0
        xi <- [...]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for sphere benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Sphere"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-1000, dims), np.repeat(1000, dims)
        self.argmins = np.repeat(0, dims).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        return lambda X :np.sum (np.square(X), axis=1)

    @property
    def objder (self) :
        """ Derivation of 1-D function """
        return lambda x : 2*x


class Matyas (Bench) :
    """
    Matyas -
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
        self.argmins = np.array([[0, 0]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return 0.26*np.sum(np.square(xy), axis=1) - 0.48*x*y

        return f


class Bulkin (Bench) :
    """
    Bulkin -
        100sqrt(|y - 0.01x^2|) + 0.01|x + 10|
        f(-10, 1) = 0
        x,y <- [-10, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor of bulkin benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Bulkin"
        self.dims = 2
        self.llim, self.rlim = np.array([-15, -3]), np.array([-5, 3])
        self.llim, self.rlim = np.repeat(-10, 2), np.repeat(10, 2)
        self.argmins = np.array([[-10, 1]])
        self.min = 0

    @property
    def obj (self) :
        """ Objetive function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return 100*np.sqrt(np.abs(
                y - 0.01*np.square(x)
            )) + 0.01*np.abs(x + 10)

        return f


class Schaffer2 (Bench) :
    """
    Schaffer-2
        0.5 + (sin^2(x^2 - y^2) - 0.5)/(1 + 0.001(x^2 + y^2))^2
        f(0, 0) = 0
        x,y <- [-100, 100]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for schaffer2 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Schaffer-2"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-100, 2), np.repeat(100, 2)
        self.argmins = np.array([[0, 0]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x2, y2 = np.square (xy[:,0]), np.square(xy[:,1])
            return 0.5 + (np.square(np.sin(x2 - y2)) - 0.5)/np.square(1 + 0.001*(x2 + y2))

        return f


class Schaffer4 (Bench) :
    """
    Schaffer-4 -
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
        self.obj_str = "Schaffer-4"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-100, 2), np.repeat(100, 2)
        self.argmins = np.array([
            [x, y]
            for x in [0, 1.25313, -1.25313]
            for y in [0, 1.25313, -1.25313]
            if (not x and y) or (not y and x)
        ])
        self.min = 0.292579

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x2 = np.square (xy[:,0])
            y2 = np.square (xy[:,1])
            return 0.5 + (np.square(np.cos(np.sin(np.abs(x2 - y2)))) - 0.5)/\
            np.square(1 + 0.001*(x2 + y2))

        return f


class Schaffer6 (Bench) :
    """
    Schaffer-6 function -
        0.5 - (sin^2(sqrt(x^2 + y^2)) - 0.5)/(1 + 0.001(x^2 + y^2))^2
        f(0, ...) = 0
        x,y <- [-100, 100]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for schaffer4 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Schaffer-6"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-100, 2), np.repeat(100, 2)
        self.argmins = np.array([[0, 0]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            sumxy = np.square(x) + np.square(y)
            return 0.5 - (np.square(np.sin(np.sqrt(sumxy))) - 0.5)/\
            np.square(1 + 0.001*sumxy)

        return f


class Griewank (Bench) :
    """
    Greiwank -
        1/4000*sum(xi^2)- prod(cos(xi/sqrt(i))) + 1
        f(0, ...) = 0
        xi <- [-600, 600]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for greiwank benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Griewank"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-600, dims), np.repeat(600, dims)
        self.argmins = np.repeat(0, dims).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (X) :
            return 1 + np.sum(np.square(X), axis=1)/4000 - \
            np.prod(np.cos(X/np.sqrt(np.arange(1, X.shape[1]+1, 1))), axis=1)

        return f

    @property
    def objder (self) :
        """ Derivation of 1-D function """
        return lambda x : x/2000 + np.sin(x)


class Rastrigin (Bench) :
    """
    Rastrigin -
        An + sum(xi^2 - Acos(2pi*xi))
        f(0, ...) = 0
        xi <- [-5.12, 5.12]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for rastrigin benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Rastrigin"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-5.12, dims), np.repeat(5.12, dims)
        self.argmins = np.repeat(0, dims).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (X) :
            A = 10
            return A*X.shape[1] + np.sum(np.square(X) - A*np.cos(2*np.pi*X) , axis=1)

        return f

    @property
    def objder (self) :
        """ Derivation of 1-D function """
        return lambda x : 2*x + 20*np.pi*np.sin(2*np.pi*x)


class RosenbrockND (Bench) :
    """
    RosenbrockND -
        sum(100(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
        f(0, ...) = 0
        xi <- [...]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for rosenbrockND benchmark function """

        super().__init__(pso_class)
        self.obj_str = "RosenbrockND"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-1000, dims), np.repeat(1000, dims)
        self.argmins = np.repeat(1, dims).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (X) :
            return 100*np.sum(np.square(X[:,1:] - np.square(X[:,:-1])), axis=1) + \
            np.sum(np.square(X[:,:-1]-1), axis=1)

        return f


class Rosenbrock2D (Bench) :
    """
    Rosenbrock2D -
        (1-x)^2 + 100(y-x^2)^2
        f(0, ...) = 0
        xi <- [...]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for rosenbrock2D benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Rosenbrock2D"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-1000, 2), np.repeat(1000, 2)
        self.argmins = np.repeat(0, 2).reshape(1, -1)
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return np.square(1-x) + 100*np.square(y - np.square(x))

        return f


class Alpine (Bench) :
    """
    Alpine -
        prod(xi*sin(xi))
        f(7.917, ...) = 2.808^n --> for maximization
        xi <- [0, 10]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for alpine benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Alpine"
        self.dims = dims
        self.llim, self.rlim = np.repeat(0, dims), np.repeat(10, dims)
        self.argmins = np.repeat(7.917, dims).reshape(1, -1)
        self.min = 2.808**dims

    @property
    def obj (self) :
        """ Objective function """
        return lambda X : -np.prod(np.sqrt(X)*np.sin(X), axis=1)

    @property
    def objder (self) :
        """ Derivation of 1-D function """
        return lambda x : np.sin(x)/(2*np.sqrt(x)) + np.sqrt(x)*np.cos(x)


class Ackley (Bench) :
    """
    Ackley -
        -20exp(-0.2*sqrt(0.5*(x^2 + y^2)))
        f(0, 0) = 0
        x,y <- [-5, 5]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for ackley benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Ackley"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-5, 2), np.repeat(5, 2)
        self.argmins = np.array([[0, 0]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            x2, y2 = np.square(x), np.square(y)
            return np.e + 20 - 20*np.exp(-0.2*np.sqrt((x2+y2)/2)) - \
            np.exp((np.cos(2*np.pi*x) + np.cos(2*np.pi*y))/2)

        return f


class Beale (Bench) :
    """
    Beale -
        (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
        f(3, 0.5) = 0
        x,y <- [-4.5, 4.5]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for beale benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Beale"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-4.5, 2), np.repeat(4.5, 2)
        self.argmins = np.array([[3, 0.5]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return np.square(1.5 - x + x*y) + np.square(2.25 - x + x*np.square(y))\
            + np.square(2.625 - x + x*np.power(y, 3))

        return f


class Goldstein (Bench) :
    """
    Goldstein -
        [1 + (x+y+1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)]
        [30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]

        f(0, -1) = 3
        x,y <- [-2, 2]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for goldstein benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Goldstein"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-2, 2), np.repeat(2, 2)
        self.argmins = np.array([[0, -1]])
        self.min = 3

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return (1+np.square(x+y+1)*(19 - 14*x + 3*np.square(x) - 14*y + 6*x*y + 3*np.square(y)))*\
            (30 + np.square(2*x - 3*y)*(18 - 32*x + 12*np.square(x) + 48*y - 36*x*y + 27*np.square(y)))

        return f


class Booth (Bench) :
    """
    Booth -
        (x+2y-7)^2 + (2x + y - 5)^2
        f(1, 3) = 0
        x,y <- [-10, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for booth benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Booth"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-10, 2), np.repeat(10, 2)
        self.argmins = np.array([[1, 3]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return np.square(x + 2*y - 7) + np.square (2*x + y - 5)

        return f


class Eggholder (Bench) :
    """
    Eggholder -
        -(y+47)sin(sqrt|x/2 + y + 47|) - xsin(sqrt|x - y - 47|)
        f(512, 404.2319) = -959.6407
        x,y <- [-512, 512]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for eggholder benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Eggholder"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-512, 2), np.repeat(512, 2)
        self.argmins = np.array([[512, 404.2319]])
        self.min = -959.6407

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return -(
            (y+47)*np.sin(np.sqrt(np.abs(
                x/2 + y + 47
            )))\
            + x*np.sin(np.sqrt(np.abs(
                x - y - 47
            )))
            )

        return f


class Easom (Bench) :
    """
    Easom -
        -cos(x)cos(y)exp(-((x-pi)^2 + (y-pi)^2))
        f(pi, pi) = -1
        x,y <- [-100, 100]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for easom benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Easom"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-100, 2), np.repeat(100, 2)
        self.argmins = np.array([[np.pi, np.pi]])
        self.min = -1

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return -(
            np.cos(x)*np.cos(y)*np.exp(-(
                np.square(x-np.pi) + np.square(y-np.pi)
                ))
            )

        return f


class McCormick (Bench) :
    """
    McCormick -
        sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1
        f(-0.54719, -1.54719) = -1.9133
        x <- [-1.5, 4]
        y <- [-3, 4]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for mccormick benchmark function """

        super().__init__(pso_class)
        self.obj_str = "McCormick"
        self.dims = 2
        self.llim, self.rlim = np.array([-1.5, 3]), np.array([4, 4])
        self.argmins = np.array([[-0.54719, -1.54719]])
        self.min = -1.9133

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return np.sin(x+y) + np.square(x-y) - 1.5*x + 2.5*y + 1

        return f


class Styblinski (Bench) :
    """
    Styblinski -
        sum(xi^4 - 16xi^2 + 5xi)/2
        f(-2.903534, ...) = -39.16616n
        xi <- [-5, 5]
    """

    def __init__ (self, pso_class, dims, Np=25) :
        """ Constructor for styblinski benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Styblinski"
        self.dims = dims
        self.llim, self.rlim = np.repeat(-5, dims), np.repeat(5, dims)
        self.argmins = np.repeat(-2.903534, dims).reshape(1, -1)
        self.min = -39.16616*dims

    @property
    def obj (self) :
        """ Objective function """
        def f (X) :
            return np.sum(np.power(X,4) - 16*np.square(X) + 5*X, axis=1)/2

        return f


class Holdertable (Bench) :
    """
    Holdetable -
         -|sin(x)cos(y)exp|1 - sqrt(x^2+y^2)/pi||
        f(+/-8.05502, +/-9.64459) - -19.2085
        x,y <- [-10, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for holdertable benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Holdertable"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-10, 2), np.repeat(10, 2)
        self.argmins = np.array([
            [x*8.05502, y*9.64459]
            for x in [1, -1]
            for y in [1, -1]
        ])
        self.min = -19.2085

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return -np.abs(
            np.sin(x)*np.cos(y)*np.exp(np.abs(
                1 - np.sqrt(np.square(x) + np.square(y))/np.pi
            )))

        return f


class Crossintray (Bench) :
    """
    Crossintray -
        -0.0001*[|sin(x)sin(y)exp|100 - sqrt(x^2+y^2)/pi|| + 1]^0.1
        f(+/-1.34941, -1.34941) = -2.06261
        x,y <- [-10, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for crossintray benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Crossintray"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-10, 2), np.repeat(10, 2)
        self.argmins = np.array([
            [x*1.34941, y*1.34941]
            for x in [1, -1]
            for y in [1, -1]
        ])
        self.min = -2.06261

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return -0.0001*np.power(1+np.abs(
                np.sin(x)*np.sin(y)*np.exp(np.abs(
                    100 - np.sqrt(np.square(x) + np.square(y))/np.pi
                ) + 1)
            ), 0.1)

        return f


class Threehumpcamel (Bench) :
    """
    Threehumpcamel -
        2x^2 - 1.05x^4 + x^6/6 + xy + y^2
        f(0, 0) = 0
        x,y <- [-5, 5]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for threehumpcamel benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Threehumpcamel"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-5, 2), np.repeat(5, 2)
        self.argmins = np.array([[0, 0]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return 2*np.square(x) - 1.05*np.power(x,4) + np.power(x,6)/6 + x*y + np.square(y)

        return f


class Himmelblau (Bench) :
    """
    Himmelblau -
        (x^2 + y - 11)^2 + (x + y^2 - 7)^2
        f(3, 2) = 0
        f(-2.805118, 3.131312) = 0
        f(-3.779310, -3.283186) = 0
        f(3.584428, -1.848126) = 0
        x,y <- [-5, 5]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for himmelblau benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Himmelblau"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-5, 2), np.repeat(5, 2)
        self.argmins = np.array([
            [3, 2],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126]
        ])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return np.square(np.square(x) + y - 11) + np.square(x + np.square(y) - 7)


        return f


class Levi (Bench) :
    """
    Levi -
        sin^2(3pi*x) + (x-1)^2(1 + sin^2(3pi*y)) + (y-1)^2(1+sin^2(2pi*y))
        f(1, 1) = 0
        x,y <- [-10, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for himmelblau benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Levi"
        self.dims = 2
        self.llim, self.rlim = np.repeat(-10, 2), np.repeat(10, 2)
        self.argmins = np.array([[1, 1]])
        self.min = 0

    @property
    def obj (self) :
        """ Objective function """
        def f (xy) :
            x, y = xy[:,0], xy[:,1]
            return np.square(np.sin(3*np.pi*x)) + np.square(x-1)*(1+np.square(np.sin(3*np.pi*y))) + \
            np.square(y-1)*(1+np.square(np.sin(2*np.pi*y)))


        return f


class Anuwu (Bench) :
    """
    Anuwu -
        0.025x^2 + sin(x)
        f(-1.49593) = -0.941254
        x <- [-20, 20]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for anuwu benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Anuwu"
        self.dims = 1
        self.llim, self.rlim = np.array([-20]), np.array([20])
        self.argmins = np.array([[-1.49593]])
        self.min = -0.941254

    @property
    def obj (self) :
        """ Objective function """
        return lambda x : 0.025*np.square(x) + np.sin(x)

    @property
    def objder (self) :
        """ Derivative of 1-D function """
        return lambda x : 0.05*x + np.cos(x)


class Ada1 (Bench) :
    """
    Ada1 -
        -(3x^5 - x^10)
        f(1.0844717712) = -9/4
        x <- [-20, 20]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for ada1 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Ada1"
        self.dims = 1
        self.llim, self.rlim = np.array([-20]), np.array([20])
        self.argmins = np.array([[1.0844717712]])
        self.min = -9/4

    @property
    def obj (self) :
        """ Objective function """
        return lambda x : -(3*np.power(x,5) - np.power(x,10))

    @property
    def objder (self) :
        """ Derivative of 1-D function """
        return lambda x : -(15*np.power(x, 4) - 10*np.power(x, 9))


class Ada2 (Bench) :
    """
    Ada2 -
        x^3 - 3x^2 + 7
        f(2) = 3
        x <- [0, 10]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for ada2 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Ada2"
        self.dims = 1
        self.llim, self.rlim = np.array([0]), np.array([10])
        self.argmins = np.array([[2]])
        self.min = 3

    @property
    def obj (self) :
        """ Objective function """
        return lambda x : np.power(x,3) - 3*np.power(x,2) + 7

    @property
    def objder (self) :
        """ Derivative of 1-D function """
        return lambda x : 3*np.square(x) - 6*x


class Ada3 (Bench) :
    """
    Ada3 -
        -exp(cos(x^2)) + x^2
        f(0) = -2.71828
        x <- [-1, 1]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for ada3 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Ada3"
        self.dims = 1
        self.llim, self.rlim = np.array([-1]), np.array([1])
        self.argmins = np.array([[0]])
        self.min = -np.e

    @property
    def obj (self) :
        """ Objective function """
        return lambda x : (lambda x2 : -np.exp(np.cos(x2)) + x2)(np.square(x))

    @property
    def objder (self) :
        """ Derivative of 1-D function """
        return lambda x : (
            lambda x, x2 : 2*np.exp(np.cos(x2))*np.sin(x2)*x + 2*x
        )(x, np.square(x))


class Ada4 (Bench) :
    """
    Ada3 -
        x^15 - sin(x) + exp(x^6)
        f(0.651208) = 0.4747057
        x <- [0, 1]
    """

    def __init__ (self, pso_class, Np=25) :
        """ Constructor for ada4 benchmark function """

        super().__init__(pso_class)
        self.obj_str = "Ada4"
        self.dims = 1
        self.llim, self.rlim = np.array([0]), np.array([1])
        self.argmins = np.array([[0.651208]])
        self.min = 0.4747057

    @property
    def obj (self) :
        """ Objective function """
        return lambda x : np.power(x,15) - np.sin(x) + np.exp(np.power(x,6))

    @property
    def objder (self) :
        """ Derivative of 1-D function """
        return lambda x : 15*np.power(x, 14) - np.cos(x) + 6*np.exp(np.power(x, 6))*np.power(x, 5)


all_benches = {
    'matyas'            : lambda p              : Matyas(p),
    'bulkin'            : lambda p              : Bulkin(p),
    'schaffer2'         : lambda p              : Schaffer2(p),
    'schaffer4'         : lambda p              : Schaffer4(p),
    'schaffer6'         : lambda p              : Schaffer6(p),
    'ackley'            : lambda p              : Ackley(p),
    'beale'             : lambda p              : Beale(p),
    'goldstein'         : lambda p              : Goldstein(p),
    'booth'             : lambda p              : Booth(p),
    'eggholder'         : lambda p              : Eggholder(p),
    'easom'             : lambda p              : Easom(p),
    'mccormick'         : lambda p              : McCormick(p),
    'holdertable'       : lambda p              : Holdertable(p),
    'crossintray'       : lambda p              : Crossintray(p),
    'threehumpcamel'    : lambda p              : Threehumpcamel(p),
    'himmelblau'        : lambda p              : Himmelblau(p),
    'levi'              : lambda p              : Levi(p),
    'rastrigin2D'       : lambda p              : Rastrigin(p, 2),
    'rosenbrock2D'      : lambda p              : Rosenbrock2D(p),
    'sphere'            : lambda p, dims=5      : Sphere(p, dims),
    'griewank'          : lambda p, dims=5      : Griewank(p, dims),
    'rastrigin'         : lambda p, dims=5      : Rastrigin(p, dims),
    'rosenbrockND'      : lambda p, dims=5      : RosenbrockND(p, dims),
    'alpine'            : lambda p, dims=5      : Alpine(p, dims),
    'styblinski'        : lambda p, dims=5      : Styblinski(p, dims)
}

test_benches = {
    'matyas'            : lambda p              : Matyas(p),
    'bulkin'            : lambda p              : Bulkin(p),
    'ackley'            : lambda p              : Ackley(p),
    'beale'             : lambda p              : Beale(p),
    'goldstein'         : lambda p              : Goldstein(p),
    'booth'             : lambda p              : Booth(p),
    'easom'             : lambda p              : Easom(p),
    'crossintray'       : lambda p              : Crossintray(p),
    'threehumpcamel'    : lambda p              : Threehumpcamel(p),
    'himmelblau'        : lambda p              : Himmelblau(p),
    'levi'              : lambda p              : Levi(p),
    'rosenbrock2D'      : lambda p              : Rosenbrock2D(p),
    'rastrigin2D'       : lambda p              : Rastrigin(p, 2),
    'griewank'          : lambda p, dims=5      : Griewank(p, dims),
    'rastrigin'         : lambda p, dims=5      : Rastrigin(p, dims),
    'rosenbrockND'      : lambda p, dims=5      : RosenbrockND(p, dims),
    'alpine'            : lambda p, dims=5      : Alpine(p, dims),
}
