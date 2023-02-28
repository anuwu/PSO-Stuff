import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pso
import cpso
import rpso
import benchmark as bm

optimizers = {
    'vanilla'       : pso.PSO,
    'adaswarm'      : cpso.EMPSO,
    'pwlc'          : cpso.PWLC_PSO,
    'rilcpso'       : rpso.RILC_PSO
}

rilcpso_vars = {
    'Og'        : rpso.RILC_PSO,
    'Var1'      : rpso.RILC_PSO_Var1,
    'Var2'      : rpso.RILC_PSO_Var2,
    'Var3'      : rpso.RILC_PSO_Var3,
    'Var4'      : rpso.RILC_PSO_Var4,
    'Var5'      : rpso.RILC_PSO_Var5
}

pwlcpso_vars = {
    'velocity'      : cpso.PWLC_PSO,
    'momentum'      : cpso.PWLC_EMPSO
}

class Suite () :
    """
    Takes in a list of pso variants and benchmarks them
    against a variety of toy functions
    """

    def __check_env__ (self) :
        """ Creates the folder environment where the benchmark
        results would be stored """

        if not os.path.isdir("Results") :
            os.mkdir("Results")

        self.suite_fold = os.path.join("Results", self.suite_name)
        if not os.path.isdir(self.suite_fold) :
            os.mkdir(self.suite_fold)


    def __init__ (self, suite_name, pso_ids, pso_classes) :
        """
        Constructor of benchmarking suite with a list of
        PSO Optimizers ready to be intialised
        """

        self.suite_name = suite_name
        self.pso_ids = pso_ids
        self.pso_classes = pso_classes
        self.specs = {}

        self.__check_env__ ()


    def eval (self, bench_iters=50, print_iters=False) :
        """ Performs the suite of benchmarks """

        df_keys = [
        'mean_iters',
        'mean_minima',
        'mean_mean_fitness',
        'mean_no_conv',
        'mean_min_err',
        'mean_argmin_err',
        'std_iters',
        'std_minima',
        'succ_ratio'
        ]

        df_col = ['pso_type'] + df_keys

        for bname, bench in bm.test_benches.items() :
            spec_csv = os.path.join(self.suite_fold, f"{bname}_ospec.csv")
            conv_png = os.path.join(self.suite_fold, f"{bname}_conv.png")
            if os.path.exists(spec_csv) and os.path.exists(conv_png) :
                continue

            print(f"On bench {bname}")
            self.specs[bname] = {}
            for pi, pc in zip(self.pso_ids, self.pso_classes) :
                print(f"Marking {pi}")
                self.specs[bname][pi] = bench(pc).eval(bench_iters, print_iters=print_iters)

            # Saving ospecs to csv
            bname_dat = [
            [pso_id] + [spec['ospec'][k] for k in df_keys]
            for pso_id, spec in self.specs[bname].items()
            ]
            df = pd.DataFrame(bname_dat, columns=df_col)
            df.to_csv(spec_csv, index=False)

            fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios' : [1, 1]})
            fig.set_figheight(5)
            fig.set_figwidth(10)

            # Saving best convergence curve
            for pi in self.pso_ids :
                ax[0].plot(self.specs[bname][pi]['ospec']['conv_curves'][0][:50], label=pi)
                ax[1].plot(self.specs[bname][pi]['ospec']['conv_curves'][1][:50], label=pi)

            # Titles, legend and saving figure to disk
            ax[0].set_title('Best Convergence')
            ax[1].set_title('Worst Convergence')
            ax[0].legend()
            ax[1].legend()
            fig.savefig(conv_png)
            plt.close()

            print("\n", end="")
