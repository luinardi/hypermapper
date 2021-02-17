#!/usr/bin/python

import os
import sys
import warnings
from collections import OrderedDict

# ensure backward compatibility
from hypermapper import optimizer  # noqa


def chakong_haimes(X):
    """
    Compute the Chakong and Haimes two-objective function to demonstrate a two-objective optimization example.
    The value is computed as defined in https://en.wikipedia.org/wiki/Test_functions_for_optimization
    :param X: dictionary containing the input points.
    :return: the two values of the Chakong and Haimes function and the feasibility indicator.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    f1_value = 2 + (x1 - 2) * (x1 - 2) + (x2 - 1) * (x2 - 1)
    f2_value = 9 * x1 - (x2 - 1) * (x2 - 1)

    # check constraints
    g1 = x1 * x1 + x2 * x2 <= 225
    g2 = x1 - 3 * x2 + 10 <= 0
    valid = g1 and g2

    output_metrics = {}
    output_metrics["f1_value"] = f1_value
    output_metrics["f2_value"] = f2_value
    output_metrics["Valid"] = valid

    return output_metrics


def main():
    parameters_file = (
        "example_scenarios/synthetic/chakong_haimes/chakong_haimes_scenario.json"
    )
    optimizer.optimize(parameters_file, chakong_haimes)
    print("End of Chakong and Haimes.")


if __name__ == "__main__":
    main()
