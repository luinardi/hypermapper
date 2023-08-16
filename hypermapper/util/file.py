import csv
import json
import os
from typing import Dict, List, Any, Tuple

import torch
from jsonschema import Draft4Validator, validators
from pkg_resources import resource_stream

from hypermapper.param.data import DataArray
from hypermapper.param.space import Space


#####################################
# JSON
#####################################

def read_settings_file(settings_file):
    """
    Reads a json settings file and returns a settings dict.

    Input:
         - file_name:
    Returns:
        - settings dict
    """
    if not settings_file.endswith(".json"):
        _, file_extension = os.path.splitext(settings_file)
        print(
            "Error: invalid file name. \nThe input file has to be a .json file not a %s"
            % file_extension
        )
        raise SystemExit
    with open(settings_file, "r") as f:
        settings = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

    default_validating_draft4_validator = extend_with_default(Draft4Validator)
    try:
        default_validating_draft4_validator(schema).validate(settings)
    except Exception as ve:
        print("Failed to validate json:")
        # print(ve)
        raise ve

    settings["log_file"] = add_path(settings, settings["log_file"])

    return settings


def extend_with_default(validator_class):
    """
    Initialize the json schema with the default values declared in the schema.json file.

    Input:
         - validator_class:
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, sub_schema in properties.items():
            if "default" in sub_schema:
                instance.setdefault(prop, sub_schema["default"])
        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_class, {"properties": set_defaults})


def validate_json(parameters_file):
    """
    Validate a json file using Hypermapper's schema.

    Input:
         - parameters_file: json file to validate.
    Returns:
        - dictionary with the contents from the json file
    """
    filename, file_extension = os.path.splitext(parameters_file)
    assert file_extension == ".json"
    with open(parameters_file, "r") as f:
        settings = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

    default_validating_draft4_validator = extend_with_default(Draft4Validator)
    default_validating_draft4_validator(schema).validate(settings)

    return settings


#####################################
# GENERAL
#####################################

def add_path(settings: Dict, file_name: str):
    """
    Add run_directory if file_name is not an absolute path.

    Input:
         - run_directory:
         - file_name:
    Returns:
        - the correct path of file_name.
    """
    if file_name[0] == "/":
        return file_name
    else:
        if settings["run_directory"] == "":
            return str(file_name)
        else:
            return os.path.join(settings["run_directory"], file_name)


def initialize_output_data_file(settings, headers):
    """
    Set the csv file where results will be written. This method checks
    if the user defined a custom filename. If not, it returns the default.
    Important: if the file exists, it will be overwritten.

    Input:
         - given_filename: the filename given in the settings file.
         - run_directory: the directory where results will be stored.
         - application_name: the name given to the application in the settings file.
    """
    if settings["output_data_file"] == "output_samples.csv":
        settings["output_data_file"] = settings["application_name"] + "_" + settings["output_data_file"]
    settings["output_data_file"] = add_path(
        settings, settings["output_data_file"]
    )
    with open(settings["output_data_file"], "w") as f:
        w = csv.writer(f)
        w.writerow(headers)


#####################################
# SAVE AND LOAD
#####################################

def load_data_file(
        space: Space,
        data_file: str,
        selection_keys_list: list = None,
        only_valid=False,
) -> DataArray:
    """
    This function read data from a csv file.

    Input:
         - space: the Space object.
         - data_file: the csv file where the data to be loaded resides.
         - selection_keys_list: contains the key columns of the csv file to be filtered.
         - only_valid: if True, only valid points are returned.
    Returns:
        - data_array
    """

    if selection_keys_list is None:
        selection_keys_list = []

    with open(data_file, "r") as f_csv:
        data = list(csv.reader(f_csv, delimiter=","))
    data = [i for i in data if len(i) > 0]
    headers = data[0]  # The first row contains the headers
    headers = [header.strip() for header in headers]
    data = [row for idx, row in enumerate(data) if idx != 0]
    # Check correctness
    for parameter_name in space.input_output_parameter_names:
        if parameter_name not in headers:
            raise Exception(
                f"Error: when reading the input dataset file the following entry was not found in the dataset but declared as a input/output parameter: {parameter_name}"
            )

    # make sure that the values are in the correct order
    parameter_indices = [headers.index(parameter_name) for parameter_name in space.parameter_names if parameter_name in selection_keys_list or not selection_keys_list]
    parameters_array = space.convert([[row[i] for i in parameter_indices] for row in data], from_type="string", to_type="internal")

    metric_indices = [headers.index(metric_name) for metric_name in space.metric_names]
    metrics_array = torch.tensor([[float(row[i]) for i in metric_indices] for row in data], dtype=torch.float64)
    if space.enable_feasible_predictor:
        feasible_array = torch.tensor([row[headers.index(space.feasible_output_name)] == space.true_value for row in data], dtype=torch.bool)
    else:
        feasible_array = torch.Tensor()
    if "timestamp" in headers:
        timestamp_array = torch.tensor([float(row[headers.index("timestamp")]) for row in data], dtype=torch.float64)
    else:
        timestamp_array = torch.zeros(parameters_array.shape[0], dtype=torch.float64)

    data_array = DataArray(parameters_array, metrics_array, timestamp_array, feasible_array)
    # Filtering the valid rows
    if only_valid:
        data_array = data_array.get_feasible()

    return data_array


def load_data_files(
        space: Space,
        filenames: List[str],
        selection_keys_list: list = [],
        only_valid: bool = False
):
    """
    Create a new data structure that contains the merged info from all the files.

    Input:
        - space: the Space object.
        - filenames: the input files that we want to merge.
        - selection_keys_list: contains the key columns of the csv file to be returned.
        - only_valid: if True, only valid points are returned.
    Returns:
        - an array with the info in the param files merged.
    """
    arrays = [load_data_file(space, filename, selection_keys_list=selection_keys_list, only_valid=only_valid)[:-1] for filename in filenames]
    data_array = arrays[0]
    for array in arrays[1:]:
        data_array.cat(array)
    return data_array


def load_previous(space: Space, settings: Dict) -> Tuple[DataArray, Any, Any]:
    """
    Loads a data from a previous to run to be continued.

    Input:
        - space: the Space object.
        - settings: the settings dictionary.
    Returns:
        - data_array: the data array.
        - absolute_configuration_index: the number of points evaluated in the previous run.
        - beginning_of_time: the timestamp of the last point evaluated in the previous run.
    """

    if not settings["resume_optimization_file"].endswith(".csv"):
        raise Exception("Error: resume data file must be a CSV")
    if settings["resume_optimization_file"] == "output_samples.csv":
        settings["resume_optimization_file"] = settings["application_name"] + "_output_samples.csv"

    data_array = load_data_file(space, settings["resume_optimization_file"])
    absolute_configuration_index = data_array.len  # get the number of points evaluated in the previous run
    beginning_of_time = data_array.timestamp_array[-1]  # Set the timestamp back to match the previous run
    print(
        "Resumed optimization, number of samples = %d ......."
        % absolute_configuration_index
    )
    return data_array, absolute_configuration_index, beginning_of_time
