from aux.functions import branin4_function
import hypermapper
import sys
from subprocess import Popen, PIPE
import os


def branin4_cli(path):
    """
    This method implements the communication protocol between the Chakong and Haimes function and Hypermapper.
    The protocol is specified in the Hypermapper wiki and it is basically an exchange of data via
    stdin and stdout using a csv-like format.
    """
    cmd = [
        "python",
        os.path.join(f"{path}", "..", "hypermapper", "run.py"),
        os.path.join(f"{path}", "..", "tests", "aux", "branin4_scenario_cli.json"),
    ]
    print(cmd)  # Command to launch Hypermapper
    p = Popen(
        cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding="utf-8"
    )  # Create a subprocess and launch Hypermapper
    i = 0
    while p.poll() is None:  # Check if the process is still running
        request = p.stdout.readline()
        p.stdout.flush()  # The first line is the request in the form: Request #_of_evaluation_requests
        if "End of" in request:  # This means that Hypermapper ended
            print(request)
            break
        elif "warning" in request:
            continue
        print("Iteration %d" % i)
        sys.stdout.write(request)
        str_to_hypermapper = "x1,x2,x3,x4,Value\n"
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
            x3 = float(parameters_values[2])
            x4 = float(parameters_values[3])
            Value = branin4_function(
                {"x1": x1, "x2": x2, "x3": x3, "x4": x4}
            )  # Evaluate objective function
            str_to_hypermapper += (
                str(x1)
                + ","
                + str(x2)
                + ","
                + str(x3)
                + ","
                + str(x4)
                + ","
                + str(Value)
                + "\n"
            )  # Add to the reply string in a csv-style
        p.stdin.write(str_to_hypermapper)
        p.stdin.flush()  # Reply to Hypermapper with all the evaluations
        i += 1
