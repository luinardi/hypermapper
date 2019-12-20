#!/usr/bin/python
import math, sys
from subprocess import Popen, PIPE

def branin_function(x1, x2):
    """
    Compute the branin function.
    :param x1: the first input of branin.
    :param x2: the second input of branin.
    :return: the value of the braning function and the (fake) energy used to compute that function.
    """
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    y_value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s

    return y_value

def communication_protocol_branin_hypermapper():
    """
    This method implements the communication protocol between the Branin function and HyperMapper.
    The protocol is specified in the HyperMapper wiki and it is basically and exchange of data via
    stdin and stdout using a csv-like format.
    """
    cmd = ["python", "scripts/hypermapper.py", "example_scenarios/synthetic/client-server_branin/client-server_branin_scenario.json"]
    print(cmd) # Command to launch HyperMapper
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding="utf-8") # Create a subprocess and launch HyperMapper
    i = 0
    while p.poll() is None: # Check if the process is still running
        request = p.stdout.readline(); p.stdout.flush() # The first line is the request in the form: Request #_of_evaluation_requests
        if request == "": # This means that HyperMapper ended
            continue
        print("Iteration %d" %i)
        sys.stdout.write(request)
        str_to_hypermapper = "x1,x2,Value\n"
        num_of_eval_requests = int(request.split(' ')[1]) # Get the #_of_evaluation_requests
        headers = p.stdout.readline(); p.stdin.flush() # The second line contains the header in the form: x1,x2
        sys.stdout.write(headers)
        for row in range(num_of_eval_requests): # Go through the rest of the eval requests
            parameters_values = p.stdout.readline() # This is an eval request in the form: number_x1, number_x2
            sys.stdout.write(parameters_values)
            parameters_values = [x.strip() for x in parameters_values.split(',')]
            x1 = float(parameters_values[0])
            x2 = float(parameters_values[1])
            y_value = branin_function(x1, x2) # Evaluate Branin function
            str_to_hypermapper += str(x1) + "," + str(x2) + "," + str(y_value) + "\n" # Add to the reply string in a csv-style
        print(str_to_hypermapper)
        p.stdin.write(str_to_hypermapper); p.stdin.flush() # Reply to HyperMapper with all the evaluations
        i += 1

def main():
    communication_protocol_branin_hypermapper()
    print("End of Branin.")

if __name__ == "__main__":
    main()