#!/usr/bin/python
import math

import os
import sys
import warnings
from collections import OrderedDict

# ensure backward compatibility
from hypermapper import optimizer  # noqa


def dtlz1_function(X):
    """
    Compute the adapted DTLZ1 function proposed by Knowles: https://www.cs.bham.ac.uk/~jdk/parego/ParEGO-TR3.pdf.
    :param X: dictionary containing the input points.
    :return: the two values of the dtlz1 function.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    x3 = X["x3"]
    x4 = X["x4"]
    x5 = X["x5"]
    x6 = X["x6"]
    g_sum = (
        (x2 - 0.5) ** 2
        + (x3 - 0.5) ** 2
        + (x4 - 0.5) ** 2
        + (x5 - 0.5) ** 2
        + (x6 - 0.5) ** 2
    )
    g_cos = (
        math.cos(2 * math.pi * (x2 - 0.5))
        + math.cos(2 * math.pi * (x3 - 0.5))
        + math.cos(2 * math.pi * (x4 - 0.5))
        + math.cos(2 * math.pi * (x5 - 0.5))
        + math.cos(2 * math.pi * (x6 - 0.5))
    )
    g = 100 * (5 + g_sum - g_cos)
    f1_value = 0.5 * x1 * (1 + g)
    f2_value = 0.5 * (1 - x1) * (1 + g)

    output_metrics = {}
    output_metrics["f1_value"] = f1_value
    output_metrics["f2_value"] = f2_value

    return output_metrics


def main():
    parameters_file = "example_scenarios/synthetic/dtlz1/dtlz1_scenario.json"
    optimizer.optimize(parameters_file, dtlz1_function)
    print("End of DTLZ1.")


if __name__ == "__main__":
    main()
