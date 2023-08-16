#!/usr/bin/python
import math

import os
import sys
import warnings
from collections import OrderedDict
import numpy as np

# ensure backward compatibility
try:
    from hypermapper import optimizer  # noqa
except:
    sys.path.append(".")
    from hypermapper import optimizer  # noqa


def hpvm(X):
    """
    Compute the branin function given two parameters.
    The value is computed as defined in https://www.sfu.ca/~ssurjano/branin.html
    :param x1: the first input of branin.
    :param x2: the second input of branin.
    :return: the value of the branin function.
    """


    return {"ExecTime" : np.sum([float(x)**2 for x in X.values()]), "Valid" : True}

def main():
    parameters_file = "example_scenarios/synthetic/hpvm/hpvm_scenario.json"
    optimizer.optimize(parameters_file, hpvm)
    print("End")

if __name__ == "__main__":
    main()
