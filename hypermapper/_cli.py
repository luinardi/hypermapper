import math

from pkg_resources import resource_filename

from hypermapper import optimizer
from hypermapper.compute_pareto import main as compute_pareto_main
from hypermapper.optimizer import main as optimizer_main
from hypermapper.plot_hvi import main as plot_hvi_main
from hypermapper.plot_optimization_results import main as plot_optimization_results
from hypermapper.plot_pareto import main as plot_pareto_main


def _compute_pareto_cli():
    compute_pareto_main()


def _hypermapper_cli():
    optimizer_main()


def _plot_pareto_cli():
    plot_pareto_main()


def _plot_hvi_cli():
    plot_hvi_main()


def _plot_optimization_results_cli():
    plot_optimization_results()


def _branin_quick_start_cli():
    def _branin_function(X):
        """
        Compute the branin function.
        :param X: dictionary containing the input points.
        :return: the value of the branin function
        """
        x1 = X["x1"]
        x2 = X["x2"]
        a = 1.0
        b = 5.1 / (4.0 * math.pi * math.pi)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)

        y_value = (
            a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
        )

        return y_value

    parameters_file = resource_filename("hypermapper", "_branin_scenario.json")
    optimizer.optimize(parameters_file, _branin_function)
    print("End of Branin.")
