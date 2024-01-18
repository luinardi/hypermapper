import os
import sys

from typing import Union, Dict, Callable, Optional

import torch

from hypermapper.util.util import (
    get_min_configurations,
    get_min_feasible_configurations,
)
from hypermapper.util.file import read_settings_file
from hypermapper.util.logging import Logger
import argparse


def optimize(
    settings_file: Union[str, Dict], black_box_function: Optional[Callable] = None
):
    """
    Optimize is the main method of Hypermapper. It takes a problem to optimize and optimization settings
    in the form of a json file or a dict adn then performs the optimization procedure.

    Input:
        - settings_file: is either a json file name or a dict
        - black_box_function: if the function to optimize is a python callable it is supplied here.
    """

    if isinstance(settings_file, str):
        settings = read_settings_file(settings_file)
    elif isinstance(settings_file, dict):
        settings = settings_file
    else:
        raise Exception(f"settings_file must be str or dict, not {type(settings_file)}")

    # INITIAL SETUP
    if not os.path.isdir(settings["run_directory"]):
        os.mkdir(settings["run_directory"])

    # set up logging
    if isinstance(sys.stdout, Logger):
        sys.stdout.change_log_file(settings["log_file"])
    else:
        sys.stdout = Logger(settings["log_file"])
    if settings["hypermapper_mode"]["mode"] == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    # print settings
    for s in settings:
        sys.stdout.write_to_logfile(s + ": " + str(settings[s]) + "\n")
    sys.stdout.write_to_logfile("\n")

    # run optimization method
    if settings["optimization_method"] in ["bayesian_optimization"]:
        from hypermapper.bo import bo

        out = bo.main(settings, black_box_function=black_box_function)
        if isinstance(out, tuple):
            return out  # configurations and parameter names
        else:
            data_array = out

    elif settings["optimization_method"] == "exhaustive":
        from hypermapper.other import exhaustive

        data_array = exhaustive.main(settings, black_box_function=black_box_function)

    else:
        print("Unrecognized optimization method:", settings["optimization_method"])
        raise SystemExit

    # If mono-objective, compute the best point found
    objectives = settings["optimization_objectives"]
    inputs = list(settings["input_parameters"].keys())
    if len(objectives) == 1:
        feasible_output = settings["feasible_output"]
        if feasible_output["enable_feasible_predictor"]:
            feasible_output_name = feasible_output["name"]
            best_point = get_min_feasible_configurations(data_array, 1)
        else:
            best_point = get_min_configurations(data_array, 1)

        keys = (
            inputs
            + objectives
            + (["feasible"] if feasible_output["enable_feasible_predictor"] else [])
        )
        best_point = (
            list(best_point.parameters_array.numpy()[0])
            + list(best_point.metrics_array.numpy()[0])
            + (
                list(best_point.feasible_array.numpy()[0])
                if feasible_output["enable_feasible_predictor"]
                else []
            )
        )
        sys.stdout.write_protocol("Best point found:\n")
        sys.stdout.write_protocol(f"{','.join(keys)}\n")
        sys.stdout.write_protocol(f"{','.join([str(v) for v in best_point])}\n\n")

    sys.stdout.write_protocol("End of HyperMapper\n")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="JSON file containing the run settings")
    args = parser.parse_args()

    if "json_file" in args:
        parameters_file = args.json_file
    else:
        print("Error: only one argument needed, the parameters json file.")

    optimize(parameters_file)


if __name__ == "__main__":
    main()
