import pso
import cpso
import ripso
import benchmark as bm

optimizers = {
    'vanilla'           : pso.PSO,
    'adaswarm'          : cpso.Adaswarm,
    'hecs'              : cpso.HECS_PSO,
    'pwlc'              : cpso.PWLC_PSO,
    'ripso'             : ripso.RI_PSO
}

class Suite () :
    """
    Takes in a list of pso variants and benchmarks them
    against a variety of toy functions
    """

    def __init__ (self, pso_classes) :
        """
        Constructor of benchmarking suite with a list of
        PSO Optimizers ready to be intialised
        """

        self.pso_classes = pso_classes
        self.specs = {}

    def eval (self, bench_iters=50) :
        """ Performs the suite of benchmarks """

        for bname, bench in bm.benches.items() :
            for pc in self.pso_classes :
                self.specs[bname] = bench(pc).eval(bench_iters, print_iters=False)
