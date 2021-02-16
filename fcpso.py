import random
import numpy as np
import pso
from math import sqrt

def constriction_vanilla(c1, c2) :
    rho = c1 + c2
    return 1 if rho <= 4 else 2.0/(2.0 - rho - sqrt(pow(rho, 2.0) - 4.0 * rho))

def constriction_em(c1, c2, beta) :
    phi = c1 + c2
    k = 4*(1 - beta)
    delta = pow(phi, 2) - k*phi

    if delta < 0 :
        return 1
    else :
        eig = (abs(phi-2) + sqrt(delta))/2
        return 1 if eig <= 1 else -1/eig


def omega_to_beta(omega) :
    return 4/omega - 1


class FCPSO (pso.PSO) :
    """
    Fairly Constricted PSO
    """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.1, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator

        The rest are defined in the base class PSO()
        """

        super().__init__(obj, llim, rlim, Np, vrat)

        # Chaotic generators for the swarm initialiser
        self.initgen = lambda x : np.random.rand(self.Np, self.D)
        self.randgen = lambda x : np.random.rand(self.Np, self.D)
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5
        self.c1_av = (self.c1_min + self.c1_max)/2
        self.c2_av = (self.c2_min + self.c2_max)/2
        self.w = 0.1

    def __str__ (self) :
        """ Optimizer descriptor """
        return "FCPSO"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        return pbest, gbest

    def optimize (self, alpha=1.2, max_iters=10000, tol=1e-2, print_iters=False) :
        """
        Performs the PSO optimization loop
        Arguments are default PSO parameters
        Returns the optimum found, and lambda function for approximate gradient
        """

        pbest, gbest = self._optim_init()

        i = 0
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1, r2 = self.randgen(1), self.randgen(2)
            c1 = random.uniform(self.c1_min, self.c1_max)
            c2 = random.uniform(self.c2_min, self.c2_max)

            # Momentum and velocity update
            self.velocity = self.w*self.velocity + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)
            self.velocity *= constriction_vanilla(c1, c2)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = self.objkey)
            self.conv_curve.append(self.objkey(gbest))

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        # Convert cache list to numpy ndarray
        grad = lambda x : -(self.c1_av*np.sum(r1) + self.c2_av*np.sum(r2))*(x - gbest)/(len(r1)*self.w)

        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)


class FCPSOem_Beta (pso.PSO) :
    """
    Fairly Constricted PSO with exponentially averaged momentum
    """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.1, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator

        The rest are defined in the base class PSO()
        """

        super().__init__(obj, llim, rlim, Np, vrat)

        # Chaotic generators for the swarm initialiser
        self.initgen = lambda x : np.random.rand(self.Np, self.D)
        self.randgen = lambda x : np.random.rand(self.Np, self.D)
        self.c1_min, self.c1_max = 1.0, 1.73360098887876
        self.c2_min, self.c2_max = 1.0, 1.73360098887876
        self.beta_min, self.beta_max = 0, 1

        self.c1_av = (self.c1_min + self.c1_max)/2
        self.c2_av = (self.c2_min + self.c2_max)/2
        self.beta_av = (self.beta_min + self.beta_max)/2

    def __str__ (self) :
        """ Optimizer descriptor """
        return "FCPSOem_Beta"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        momentum = np.zeros_like(self.particles)
        return momentum, pbest, gbest

    def optimize (self, alpha=1.2, max_iters=10000, tol=1e-2, print_iters=False) :
        """
        Performs the PSO optimization loop
        Arguments are default PSO parameters
        Returns the optimum found, and lambda function for approximate gradient
        """

        momentum, pbest, gbest = self._optim_init()

        i = 0
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1, r2 = self.randgen(1), self.randgen(2)
            c1 = random.uniform(self.c1_min, self.c1_max)
            c2 = random.uniform(self.c2_min, self.c2_max)
            beta = random.uniform(self.beta_min, self.beta_max)

            # Momentum and velocity update
            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)
            self.velocity *= constriction_em(c1, c2, beta)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = self.objkey)

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        # Convert cache list to numpy ndarray
        grad = lambda x : -(self.c1_av*np.sum(r1) + self.c2_av*np.sum(r2))*(x - gbest)/(len(r1)*(1-self.beta_av))

        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)

class FCPSOem_Omega (pso.PSO) :
    """
    Fairly Constricted PSO with exponentially averaged momentum
    """

    def __init__ (self, obj, llim, rlim, Np, vrat=0.1, cache=False) :
        """
        Constructor of the PSO Optimizer with limits and random
        number generators
            initer      - Position and velocity initialiser
            rander      - r1, r2 generator

        The rest are defined in the base class PSO()
        """

        super().__init__(obj, llim, rlim, Np, vrat)

        # Chaotic generators for the swarm initialiser
        self.initgen = lambda x : np.random.rand(self.Np, self.D)
        self.randgen = lambda x : np.random.rand(self.Np, self.D)
        self.c1_min, self.c1_max = 1.0, 2.0
        self.c2_min, self.c2_max = 1.0, 2.0
        self.omega_min, self.omega_max = 2, 4

        self.c1_av = (self.c1_min + self.c1_max)/2
        self.c2_av = (self.c2_min + self.c2_max)/2
        self.omega_av = (self.omega_min + self.omega_max)/2

    def __str__ (self) :
        """ Optimizer descriptor """
        return "FCPSOem_Omega"

    def _optim_init (self) :
        """ Initialiser of certain state variables before the optimization loop """

        pbest, gbest = super()._optim_init()
        momentum = np.zeros_like(self.particles)
        return momentum, pbest, gbest

    def optimize (self, alpha=1.2, max_iters=10000, tol=1e-2, print_iters=False) :
        """
        Performs the PSO optimization loop
        Arguments are default PSO parameters
        Returns the optimum found, and lambda function for approximate gradient
        """

        momentum, pbest, gbest = self._optim_init()

        i = 0
        while True :
            # Using the first and second internal generators, randgen(1) and radgen(2) respectively
            r1, r2 = self.randgen(1), self.randgen(2)
            c1 = random.uniform(self.c1_min, self.c1_max)
            c2 = random.uniform(self.c2_min, self.c2_max)
            omega = random.uniform(self.omega_min, self.omega_max)
            beta = omega_to_beta(omega)

            # Momentum and velocity update
            momentum = beta*momentum + (1-beta)*self.velocity
            self.velocity = momentum + c1*r1*(pbest - self.particles) + c2*r2*(gbest - self.particles)
            self.velocity *= constriction_em(c1, c2, beta)

            # Perform velocity clipping before running ipcd() to minimize any violations
            self.velocity = pso.vclip(self.velocity, self.vmax)

            ######################################################################
            # Perform "Inverse Parabolic Confined Distribution" technique for
            # boundary handling. Also returns updated particle position and velocity
            ######################################################################
            self.particles, self.velocity = pso.ipcd(self.particles, self.velocity, self.llim, self.rlim, alpha)

            # Update pbest, gbest
            less = self.obj(self.particles) < self.obj(pbest)
            pbest[less] = np.copy(self.particles[less])
            gbest = min(pbest , key = self.objkey)

            i += 1
            if print_iters : print("\r{}".format(i), end="")

            # Stopping criteria
            if i == max_iters or (np.abs(self.particles - gbest) < tol).all() :
                break

        # Convert cache list to numpy ndarray
        grad = lambda x : -(self.c1_av*np.sum(r1) + self.c2_av*np.sum(r2))*(x - gbest)/(len(r1)*(1 - omega_to_beta(self.omega_av)))

        if print_iters : print("\n", end="")
        return self.optRet(gbest, grad, tol, i)
