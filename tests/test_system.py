import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'example_scenarios/quick_start/')
sys.path.insert(0, 'example_scenarios/synthetic/ordinal_multiobjective_branin')
sys.path.insert(0, 'example_scenarios/synthetic/multiobjective_branin')
import ordinal_multiobjective_branin
import multiobjective_branin
import branin
import pytest
import compute_pareto
import plot_pareto
import hypermapper
from plot_hvi import HVI_from_files
import os
from os.path import isfile, join
from subprocess import Popen, PIPE
from utility_functions import *

def test_quick_start():
    """
    This test uses Hypermapper's quick start guide, which is the Branin function.
    The goal is to ensure Hypermapper still works on this introductory example.
    """
    parameters_file = "example_scenarios/synthetic/multiobjective_branin/multiobjective_branin_scenario.json"
    exhaustive_pareto_file = "tests/data/branin_grid_search_pareto.csv"
    validate_json(parameters_file)
    multiobjective_branin.main()
    compute_pareto.main(parameters_file)
    plot_pareto.main(parameters_file)

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
    compute_pareto.main(parameters_file)

    hvi = HVI_from_files(exhaustive_pareto_file, parameters_file)
    assert hvi < 300

def test_black_scholes():
    """
    This test uses Spatial's BlackScholes use case.
    The goal is to test performance on a simple real world use case where we can perform grid search and compute the exact Pareto front.
    """
    exhaustive_pareto_file = "tests/data/BlackScholes_real_pareto.csv"
    parameters_file = "tests/data/BlackScholes_scenario.json"
    validate_json(parameters_file)
    cmd = ["python", "scripts/hypermapper.py", parameters_file]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding="utf-8")
    p.wait()

    compute_pareto.main(parameters_file)

    hvi = HVI_from_files(exhaustive_pareto_file, parameters_file)
    assert hvi < 80000

if __name__ == '__main__':
    test_quick_start()
    test_ordinal_branin()
    test_black_scholes()
