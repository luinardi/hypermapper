#!/usr/bin/python
import math

import os
import sys
import warnings
from collections import OrderedDict

# ensure backward compatibility
from hypermapper import optimizer  # noqa


def branin_function_two_objectives(X):
    """
    Compute the branin function and also a fake energy (in Joules) function to demonstrate a two-objective optimization example.
    :param x1: the first input of branin.
    :param x2: the second input of branin.
    :return: the value of the braning function and the (fake) energy used to compute that function.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    y_value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
    y_energy = x1 + x2

    output_metrics = {}
    output_metrics["Value"] = y_value
    output_metrics["Energy"] = y_energy

    return output_metrics


def main():
    parameters_file = "example_scenarios/synthetic/multiobjective_branin/multiobjective_branin_scenario.json"
    optimizer.optimize(parameters_file, branin_function_two_objectives)
    print("End of Branin.")


if __name__ == "__main__":
    main()
