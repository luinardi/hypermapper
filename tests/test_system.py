import subprocess
import os
from subprocess import Popen, PIPE, STDOUT

from example_scenarios.quick_start.branin import main as quick_start_branin_main
from example_scenarios.synthetic.multiobjective_branin import multiobjective_branin
from example_scenarios.synthetic.ordinal_multiobjective_branin import (
    ordinal_multiobjective_branin,
)
from hypermapper import compute_pareto
from hypermapper import plot_pareto
from hypermapper.plot_hvi import HVI_from_files
from hypermapper import optimizer
from hypermapper.utility_functions import validate_json


def test_black_scholes():
    """
    This test uses Spatial's BlackScholes use case.
    The goal is to test performance on a simple real world use case where we can perform grid search and compute the exact Pareto front.
    """
    exhaustive_pareto_file = "tests/data/BlackScholes_real_pareto.csv"
    parameters_file = "tests/data/BlackScholes_scenario.json"
    validate_json(parameters_file)

    # hypermapper.optimize(parameters_file)
    cmd = ["python", "scripts/hypermapper.py", parameters_file]
    subprocess.check_call(cmd, stderr=STDOUT)

    compute_pareto.compute(parameters_file, None, None)

    hvi = HVI_from_files(exhaustive_pareto_file, parameters_file)
    assert hvi < 100000


def test_quick_start():
    """
    This test uses Hypermapper's quick start guide, which is the Branin function.
    The goal is to ensure Hypermapper still works on this introductory example.
    """
    parameters_file = "example_scenarios/synthetic/multiobjective_branin/multiobjective_branin_scenario.json"
    exhaustive_pareto_file = "tests/data/branin_grid_search_pareto.csv"
    validate_json(parameters_file)
    multiobjective_branin.main()
    compute_pareto.compute(parameters_file, None, None)
    plot_pareto.plot(parameters_file)

    hvi = HVI_from_files(exhaustive_pareto_file, parameters_file)
    assert hvi < 500


def test_ordinal_branin():
    """
    This test uses the Branin function with ordinal variables.
    The goal is to test a simple case where we know the exact Pareto front.
    """
    exhaustive_pareto_file = "tests/data/ordinal_branin_real_pareto.csv"
    parameters_file = "example_scenarios/synthetic/ordinal_multiobjective_branin/ordinal_multiobjective_branin_scenario.json"
    validate_json(parameters_file)
    ordinal_multiobjective_branin.main()
    compute_pareto.compute(parameters_file, None, None)

    hvi = HVI_from_files(exhaustive_pareto_file, parameters_file)
    assert hvi < 300


if __name__ == "__main__":
    test_black_scholes()
    test_quick_start()
    test_ordinal_branin()
