#!/usr/bin/python
import math

import os
import sys
import warnings
from collections import OrderedDict

# ensure backward compatibility
from hypermapper import optimizer  # noqa


def branin_function(x1, x2):
    """
    Compute the branin function given two parameters.
    The value is computed as defined in https://www.sfu.ca/~ssurjano/branin.html
    :param x1: the first input of branin.
    :param x2: the second input of branin.
    :return: the value of the branin function.
    """
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    y_value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s

    return y_value


def branin4_function(X):
    """
    Compute the four-dimensional branin function.
    Value in four-dimensional branin is defined as the product of two Brannin functions.
    :param X: dictionary containing the input points.
    :return: the value of the branin function.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    x3 = X["x3"]
    x4 = X["x4"]
    f1_value = branin_function(x1, x2)
    f2_value = branin_function(x3, x4)
    y_value = f1_value * f2_value

    return y_value


def main():
    parameters_file = "example_scenarios/synthetic/branin4/branin4_scenario.json"
    optimizer.optimize(parameters_file, branin4_function)
    print("End of Branin4.")


if __name__ == "__main__":
    main()
