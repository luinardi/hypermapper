#!/usr/bin/python
import math
import numpy as np
import os
import sys
import warnings
from collections import OrderedDict


sys.path.append(".")
from hypermapper import optimizer  # noqa


def taco(X):
    x1 = X["chunk_size"]
    x2 = X["omp_chunk_size"]
    x3 = X["omp_num_threads"]
    x4 = X["omp_scheduling_type"]
    p = X["permutation"]
    x5 = X["unroll_factor"]

    return {
        "compute_time": x1**2
        + x2**2
        + x3**2
        + x4**2
        + x5**2
        + p[0] ** 2
        + p[1]
        - p[3]
        - p[4] ** 2
        + 2 ** np.random.normal(),
        "Valid": True,
    }


def main():
    parameters_file = "example_scenarios/synthetic/taco/taco.json"
    optimizer.optimize(parameters_file, taco)
    print("End of known constraints")


if __name__ == "__main__":
    main()
