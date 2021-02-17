#!/usr/bin/python

import os
import sys
import warnings
from collections import OrderedDict

from hypermapper import optimizer  # noqa

from subprocess import Popen, PIPE


def chakong_haimes(x1, x2):
    """
    Compute the Chakong and Haimes two-objective function to demonstrate a two-objective optimization example.
    The value is computed as defined in https://en.wikipedia.org/wiki/Test_functions_for_optimization
    :param x1: the first input of the function.
    :param x2: the second input of function.
    :return: the two values of the Chakong and Haimes function and the feasibility indicator.
    """
    f1_value = 2 + (x1 - 2) * (x1 - 2) + (x2 - 1) * (x2 - 1)
    f2_value = 9 * x1 - (x2 - 1) * (x2 - 1)

    # check constraints
    g1 = x1 * x1 + x2 * x2 <= 225
    g2 = x1 - 3 * x2 + 10 <= 0
    valid = g1 and g2

    return f1_value, f2_value, valid


def communication_protocol_chakong_haimes_hypermapper():
    """
    This method implements the communication protocol between the Chakong and Haimes function and HyperMapper.
    The protocol is specified in the HyperMapper wiki and it is basically an exchange of data via
    stdin and stdout using a csv-like format.
    """
    cmd = [
        "python",
        "scripts/hypermapper.py",
        "example_scenarios/clients/python/client-server_chakong_haimes_scenario.json",
    ]
    print(cmd)  # Command to launch HyperMapper
    p = Popen(
        cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding="utf-8"
    )  # Create a subprocess and launch HyperMapper
    i = 0
    while p.poll() is None:  # Check if the process is still running
        request = p.stdout.readline()
        p.stdout.flush()  # The first line is the request in the form: Request #_of_evaluation_requests
        if "End of HyperMapper" in request:  # This means that HyperMapper ended
            print(request)
            break
        elif "warning" in request:
            continue
        print("Iteration %d" % i)
        sys.stdout.write(request)
        str_to_hypermapper = "x1,x2,f1_value,f2_value,Valid\n"
        num_of_eval_requests = int(
            request.split(" ")[1]
        )  # Get the #_of_evaluation_requests
        headers = p.stdout.readline()
        p.stdin.flush()  # The second line contains the header in the form: x1,x2
        sys.stdout.write(headers)
        for row in range(
            num_of_eval_requests
        ):  # Go through the rest of the eval requests
            parameters_values = (
                p.stdout.readline()
            )  # This is an eval request in the form: number_x1, number_x2
            sys.stdout.write(parameters_values)
            parameters_values = [x.strip() for x in parameters_values.split(",")]
            x1 = float(parameters_values[0])
            x2 = float(parameters_values[1])
            f1_value, f2_value, valid = chakong_haimes(
                x1, x2
            )  # Evaluate objective function
            str_to_hypermapper += (
                str(x1)
                + ","
                + str(x2)
                + ","
                + str(f1_value)
                + ","
                + str(f2_value)
                + ","
                + str(valid)
                + "\n"
            )  # Add to the reply string in a csv-style
        print(str_to_hypermapper)
        p.stdin.write(str_to_hypermapper)
        p.stdin.flush()  # Reply to HyperMapper with all the evaluations
        i += 1


def main():
    communication_protocol_chakong_haimes_hypermapper()
    print("End of Chakong and Haimes.")


if __name__ == "__main__":
    main()
