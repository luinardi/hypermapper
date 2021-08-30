import json
import os
import sys
import warnings
from collections import OrderedDict

from jsonschema import exceptions, Draft4Validator
from pkg_resources import resource_stream

import numpy as np

# ensure backward compatibility
try:
    from hypermapper import bo
    from hypermapper import evolution
    from hypermapper import local_search
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        Logger,
        extend_with_default,
        get_min_configurations,
        get_min_feasible_configurations,
    )
    from hypermapper.profiling import Profiler
except ImportError:
    if os.getenv("HYPERMAPPER_HOME"):  # noqa
        warnings.warn(
            "Found environment variable 'HYPERMAPPER_HOME', used to update the system path. Support might be discontinued in the future. Please make sure your installation is working without this environment variable, e.g., by installing with 'pip install hypermapper'.",
            DeprecationWarning,
            2,
        )  # noqa
        sys.path.append(os.environ["HYPERMAPPER_HOME"])  # noqa
    ppath = os.getenv("PYTHONPATH")
    if ppath:
        path_items = ppath.split(":")

        scripts_path = ["hypermapper/scripts", "hypermapper_dev/scripts"]

        if os.getenv("HYPERMAPPER_HOME"):
            scripts_path.append(os.path.join(os.getenv("HYPERMAPPER_HOME"), "scripts"))

        truncated_items = [
            p for p in sys.path if len([q for q in scripts_path if q in p]) == 0
        ]
        if len(truncated_items) < len(sys.path):
            warnings.warn(
                "Found hypermapper in PYTHONPATH. Usage is deprecated and might break things. "
                "Please remove all hypermapper references from PYTHONPATH. Trying to import"
                "without hypermapper in PYTHONPATH..."
            )
            sys.path = truncated_items

    sys.path.append(".")  # noqa
    sys.path = list(OrderedDict.fromkeys(sys.path))

    from hypermapper import bo
    from hypermapper import evolution
    from hypermapper import local_search
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        Logger,
        extend_with_default,
        get_min_configurations,
        get_min_feasible_configurations,
    )
    from hypermapper.profiling import Profiler


def optimize(parameters_file, black_box_function=None):
    try:
        hypermapper_pwd = os.environ["PWD"]
        hypermapper_home = os.environ["HYPERMAPPER_HOME"]
        os.chdir(hypermapper_home)
        warnings.warn(
            "Found environment variable 'HYPERMAPPER_HOME', used to update the system path. Support might be discontinued in the future. Please make sure your installation is working without this environment variable, e.g., by installing with 'pip install hypermapper'.",
            DeprecationWarning,
            2,
        )
    except:
        hypermapper_pwd = "."

    if not parameters_file.endswith(".json"):
        _, file_extension = os.path.splitext(parameters_file)
        print(
            "Error: invalid file name. \nThe input file has to be a .json file not a %s"
            % file_extension
        )
        raise SystemExit
    with open(parameters_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

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
    log_file = deal_with_relative_and_absolute_path(run_directory, log_file)
    sys.stdout = Logger(log_file)

    optimization_method = config["optimization_method"]
    if config["profiling"]:
        profiling = Profiler(config)
        profiling.run()
    else:
        profiling = None

    if (
        (optimization_method == "random_scalarizations")
        or (optimization_method == "bayesian_optimization")
        or (optimization_method == "prior_guided_optimization")
    ):
        data_array = bo.main(
            config, black_box_function=black_box_function, profiling=profiling
        )
    elif optimization_method == "local_search":
        data_array = local_search.main(
            config, black_box_function=black_box_function, profiling=profiling
        )
    elif optimization_method == "evolutionary_optimization":
        data_array = evolution.main(
            config, black_box_function=black_box_function, profiling=profiling
        )
    else:
        print("Unrecognized optimization method:", optimization_method)
        raise SystemExit
    if config["profiling"]:
        profiling.stop()

    try:
        os.chdir(hypermapper_pwd)
    except:
        pass

    # If mono-objective, compute the best point found
    objectives = config["optimization_objectives"]
    inputs = list(config["input_parameters"].keys())
    if len(objectives) == 1:
        explored_points = {}
        for parameter in inputs + objectives:
            explored_points[parameter] = data_array[parameter]
        objective = objectives[0]
        feasible_output = config["feasible_output"]
        if feasible_output["enable_feasible_predictor"]:
            feasible_parameter = feasible_output["name"]
            explored_points[feasible_parameter] = data_array[feasible_parameter]
            best_point = get_min_feasible_configurations(
                explored_points, 1, objective, feasible_parameter
            )
        else:
            best_point = get_min_configurations(explored_points, 1, objective)
        keys = ""
        best_point_string = ""
        for parameter in inputs + objectives:
            keys += f"{parameter},"
            best_point_string += f"{best_point[parameter][0]},"
        keys = keys[:-1]
        best_point_string = best_point_string[:-1]

    # If there is a best point, return it according the user's preference
    print_best = config["print_best"]
    if (print_best is not True) and (print_best is not False):
        if print_best != "auto":
            print(
                f"Warning: unrecognized option for print_best: {print_best}. Should be either 'auto' or a boolean."
            )
            print("Using default.")
        hypermapper_mode = config["hypermapper_mode"]["mode"]
        print_best = False if hypermapper_mode == "client-server" else True

    if print_best:
        if len(objectives) == 1:
            sys.stdout.write_protocol("Best point found:\n")
            sys.stdout.write_protocol(f"{keys}\n")
            sys.stdout.write_protocol(f"{best_point_string}\n\n")
        else:
            if (
                config["print_best"] is True
            ):  # If the user requested this, let them know it is not possible
                sys.stdout.write_protocol(
                    "\nMultiple objectives, there is no single best point.\n"
                )
    else:
        if len(objectives) > 1:
            sys.stdout.write_to_logfile(
                "\nMultiple objectives, there is no single best point.\n"
            )
        else:
            sys.stdout.write_to_logfile("Best point found:\n")
            sys.stdout.write_to_logfile(f"{keys}\n")
            sys.stdout.write_to_logfile(f"{best_point}\n\n")

    sys.stdout.write_protocol("End of HyperMapper\n")


def main():
    if len(sys.argv) == 2:
        parameters_file = sys.argv[1]
    else:
        print("Error: only one argument needed, the parameters json file.")

    if parameters_file == "--help" or len(sys.argv) != 2:
        print("#########################################")
        print("HyperMapper: a multi-objective black-box optimization tool")
        print(
            "Quickstart guide: https://github.com/luinardi/hypermapper/wiki/Quick-Start-Guide"
        )
        print("Full documentation: https://github.com/luinardi/hypermapper/wiki")
        print("Useful commands:")
        print(
            "    hm-quickstart                                                            test the installation with a quick optimization run"
        )
        print(
            "    hypermapper /path/to/configuration_file                                  run HyperMapper in client-server mode"
        )
        print(
            "    hm-plot-optimization-results /path/to/configuration_file                 plot the results of a mono-objective optimization run"
        )
        print(
            "    hm-compute-pareto /path/to/configuration_file                            compute the pareto of a two-objective optimization run"
        )
        print(
            "    hm-plot-pareto /path/to/configuration_file /path/to/configuration_file   plot the pareto computed by hm-compute-pareto"
        )
        print(
            "    hm-plot-hvi /path/to/configuration_file /path/to/configuration_file      plot the hypervolume indicator for a multi-objective optimization run"
        )
        print("###########################################")
        exit(1)

    optimize(parameters_file)


if __name__ == "__main__":
    main()
