import sys
import random_scalarizations
import local_search
import compute_pareto
import plot_dse
import json
import os
from utility_functions import *
import json
from jsonschema import Draft4Validator, validators, exceptions


def optimize(parameters_file, black_box_function=None):

    try:
        hypermapper_pwd = os.environ['PWD']
        hypermapper_home = os.environ['HYPERMAPPER_HOME']
        os.chdir(hypermapper_home)
        #print(hypermapper_home); print(hypermapper_pwd)
    except:
        #print("HYPERMAPPER_HOME environment variable not found, set to '.'")
        hypermapper_home = "."
        hypermapper_pwd = "."

    if not parameters_file.endswith('.json'):
        _, file_extension = os.path.splitext(parameters_file)
        print("Error: invalid file name. \nThe input file has to be a .json file not a %s" %file_extension)
        raise SystemExit
    with open(parameters_file, 'r') as f:
        config = json.load(f)

    json_schema_file = 'scripts/schema.json'
    with open(json_schema_file, 'r') as f:
        schema = json.load(f)

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    try:
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the client-server mode we only want to log on the file.
    run_directory = config["run_directory"]
    if run_directory == ".":
        run_directory = hypermapper_pwd
        config["run_directory"] = run_directory
    log_file = config["log_file"]
    if log_file == "hypermapper_logfile.log":
        log_file = deal_with_relative_and_absolute_path(run_directory, log_file)
    sys.stdout = Logger(log_file)

    optimization_method = config["optimization_method"]

    if optimization_method == "random_scalarizations":
        random_scalarizations.main(config, black_box_function=black_box_function)
    elif optimization_method == "local_search":
        local_search.main(config, black_box_function=black_box_function)
    else:
        print("Unrecognized optimization method:", optimization_method)
        raise SystemExit

    try:
        os.chdir(hypermapper_pwd)
    except:
        pass

    print("### End of the hypermapper script.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        parameters_file = sys.argv[1]
    else :
        print("Error: only one argument needed, the parameters json file.")

    if parameters_file == "--help" or len(sys.argv) != 2:
        print("################################################")
        print("### Example: ")
        print("### cd hypermapper")
        print("### python3 scripts/hypermapper.py example_scenarios/spatial/BlackScholes_scenario.json")
        print("################################################")
        exit(1)

    optimize(parameters_file)