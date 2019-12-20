#!/usr/bin/python
import math, sys
from subprocess import Popen, PIPE
sys.path.append('scripts')
import hypermapper

def currin_exp_function(X):
    """
    Compute the CurrinExp function.
    The value is computed as defined in https://www.sfu.ca/~ssurjano/curretal88exp.html
    :param X: dictionary containing the input points.
    :return: the value of the CurrinExp function.
    """
    x1 = X['x1']
    x2 = X['x2']
    factor1 = 1 - math.exp(-1/(2*x2))
    factor2 = 2300*x1*x1*x1 + 1900*x1*x1 + 2092*x1 + 60
    factor3 = 100*x1*x1*x1 + 500*x1*x1 + 4*x1 + 20
    y_value = factor1*factor2/factor3

    return y_value

def main():
    parameters_file = "example_scenarios/synthetic/currinexp/currinexp_scenario.json"
    hypermapper.optimize(parameters_file, currin_exp_function)
    print("End of CurrinExp.")

if __name__ == "__main__":
    main()