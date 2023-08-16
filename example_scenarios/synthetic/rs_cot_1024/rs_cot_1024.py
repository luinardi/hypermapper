#!/usr/bin/python
import math

import os
import sys
import warnings
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from hypermapper import optimizer  # noqa
except:
    sys.path.append(".")
    from hypermapper import optimizer  # noqa


def rs_cot_1024(X):

    if isinstance(X, dict):
        v3 = X["tuned_v3"]
        v4 = X["tuned_v4"]
        v5 = X["tuned_v5"]
        v6 = X["tuned_v6"]
        v7 = X["tuned_v7"]
        v8 = X["tuned_v8"]

        ls0 = X["tuned_ls0"]
        ls1 = X["tuned_ls1"]
        gs0 = X["tuned_gs0"]
        gs1 = X["tuned_gs1"]

    else:
        v3, v4, v5, v6, v7, v8, ls0, ls1, gs0, gs1 = tuple(X)

    alpha = 0.00001
    beta = 800
    gamma = 10
    delta = 100

    # dummy computation for function value
    function_value = (
        alpha * v3 * v4  # 0 - 1e2
        + alpha * (v5 + v6 - beta) ** 2  # 0 - 1e2
        - gamma * np.sin(v7 / delta) * np.sin(v8 / delta)  # -1e1 - 1e1
        + np.log(ls0 + 1)
        - np.log(ls1 + 1)  # 0 - 1e1
        + np.log(gs0 + 1)
        - np.log(gs1 + 1)  # -1e1 - 0
        + alpha * (v3 + v5 + v7 - ls0 - gs1) ** 2  # 0 - 1e2
    )

    valid = True
    if v3 <= 16:
        valid = False
        function_value = 100000
    function_value = np.float64(function_value)
    return {"runtime": function_value, "Valid": True}


def main():
    parameters_file = (
        "example_scenarios/synthetic/rs_cot_1024/rs_cot_1024_scenario.json"
    )
    optimizer.optimize(parameters_file, rs_cot_1024)
    print("End of known constraints")


if __name__ == "__main__":
    main()
