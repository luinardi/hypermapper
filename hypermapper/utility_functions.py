import copy
import json
import multiprocessing as mp
import os
import sys
import warnings
from collections import defaultdict, OrderedDict
import csv

import numpy as np
from jsonschema import Draft4Validator, validators
from pkg_resources import resource_stream
from scipy import stats

# ensure backward compatibility
try:
    from hypermapper import optimizer
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


####################################################
# Files and paths management
####################################################


def get_last_dir_and_file_names(file_path):
    """
    Return a string with the name of the last directory and the name of the file in file_path.
    :param file_path: Example: /dir1/dir2/filename.txt
    :return: a string. Example: dir2/filename.txt
    """
    path = os.path.dirname(file_path)
    dir = os.path.basename(path)
    file = os.path.basename(file_path)
    return str(dir + "/" + file)


def get_path_and_file_name_without_extension(file_path):
    """
    Return a string with the path and the name of the file in file_path without extension.
    :param file_path: Example: /dir1/dir2/filename.txt
    :return: a string. Example: /dir1/dir2/filename
    """
    path = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    file = os.path.splitext(file)[0]
    return os.path.join(str(dir), file)


def deal_with_relative_and_absolute_path(run_directory, file_name):
    """
    Add run_directory if file_name is not an absolute path.
    :param run_directory:
    :param file_name:
    :return: the correct path of file_name.
    """
    if file_name[0] == "/":
        return file_name
    else:
        if run_directory == "":
            return str(file_name)
        else:
            return os.path.join(run_directory, file_name)


def get_output_data_file(given_filename, run_directory, application_name):
    """
    Get the csv file where results will be written. This method checks
    if the user defined a custom filename. If not, it returns the default.
    Important: if the file exists, it will be overwritten.
    :param given_filename: the filename given in the configuration file.
    :param run_directory: the directory where results will be stored.
    :param application_name: the name given to the application in the configuration file.
    """
    output_data_file = given_filename
    if output_data_file == "output_samples.csv":
        output_data_file = application_name + "_" + output_data_file
    output_data_file = deal_with_relative_and_absolute_path(
        run_directory, output_data_file
    )
    return output_data_file


####################################################
# Logging
####################################################
class Logger:
    """
    This class allows to write on the log file and to the stdout at the same time.
    In HyperMapper the log is always created. The stdout can be switch off in interactive mode for example.
    It works overloading sys.stdout.
    """

    def __init__(self, log_file="hypermapper_logfile.log"):
        self.filename = log_file
        self.terminal = sys.stdout
        try:
            self.log = open(self.filename, "a")
        except:
            print("Unexpected error opening the log file: ", self.filename)
            raise
        self.log_only_on_file = False
        self.is_verbose = False

    def write(self, message):
        if not self.log_only_on_file:
            self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def write_protocol(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush_protocol()

    def flush(self):
        if not self.log_only_on_file:
            self.terminal.flush()
        self.log.flush()

    def flush_protocol(self):
        self.terminal.flush()
        self.log.flush()

    def switch_log_only_on_file(self, choice):
        self.log_only_on_file = choice

    def change_log_file(self, filename):
        if self.filename != filename:
            self.close_log_file()
            try:
                self.log = open(self.filename, "a")
            except:
                print("Unexpected error opening the log file: ", self.filename)
                raise

    def set_verbose_mode(self, is_verbose):
        self.is_verbose = is_verbose

    def write_to_logfile(self, message, msg_is_verbose=False):
        if not msg_is_verbose or self.is_verbose:
            self.log.write(message)

    def close_log_file(self):
        self.log.close()

    def __del__(self):
        try:
            self.log.close()
        except:
            print("Warning: exception raised closing the log file: ", self.filename)


####################################################
# Output writers
####################################################


def create_output_data_file(filename, headers):
    """
    Create a csv file and write the optimization headers to it.
    If a filename is not given, the output_data_file given in the json will be used.
    Important: if the file exists, it will be overwritten.
    :param data_array: the data array to write
    :param filename: the file where data will be written
    """
    with open(filename, "w") as f:
        w = csv.writer(f)
        w.writerow(headers)


def write_data_array(param_space, data_array, filename):
    """
    Write a data array to a csv file.
    If a filename is not given, the output_data_file given in the json will be used.
    If the file does not exist, it will be created.
    :param data_array: the data array to write
    :param filename: the file where data will be written
    """
    if not os.path.isfile(filename):
        create_output_data_file(
            filename, param_space.get_input_output_and_timestamp_parameters()
        )

    with open(filename, "a") as f:
        w = csv.writer(f)
        tmp_list = [
            param_space.convert_types_to_string(j, data_array)
            for j in list(param_space.get_input_output_and_timestamp_parameters())
        ]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(data_array[list(data_array.keys())[0]])):
            w.writerow(tmp_list[i])


####################################################
# Data structure handling
####################################################
def concatenate_data_dictionaries(D1, D2, selection_keys_list=[]):
    """
    Concatenate dictionaries.
    :param D1: first dictionary.
    :param D2: second dictionary.
    :return: the concatenated dictionaries.
    """
    D3 = {}
    if len(selection_keys_list) == 0:
        keys = set(list(D1.keys()) + list(D2.keys()))
    else:
        keys = set(selection_keys_list)

    for e in keys:
        if len(D1) > 0 and len(D2) > 0:
            D3[e] = D1[e][:] + D2[e][:]
        elif len(D1) == 0:
            D3[e] = D2[e][:]
        else:
            D3[e] = D1[e][:]

    return D3


def concatenate_list_of_dictionaries(dict_list, selection_keys_list=None):
    concatenated_dict = defaultdict(list)

    if selection_keys_list is None:
        selection_keys_list = list(dict_list[0].keys())

    for D in dict_list:
        for key in selection_keys_list:
            concatenated_dict[key].append(D[key])
    return concatenated_dict


def get_single_configuration(configurations, idx):
    """
    Get a single configuration dictionary from a dictionary containing multiple configurations.
    :param configurations: a dictionary of lists containing multiple configurations.
    :param idx: the index of the desired configuration.
    :return: dictionary containing a single configuration.
    """
    single_configuration = {}
    for key in configurations:
        single_configuration[key] = configurations[key][idx]
    return single_configuration


def are_configurations_equal(configuration1, configuration2, keys):
    """
    Compare two configurations. They are considered equal if they hold the same values for all keys.
    :param configuration1: the first configuration in the comparison
    :param configuration2: the second configuration in the comparison
    :param keys: the keys to use for comparison
    :return: boolean indicating if configurations are equal or not
    """
    for key in keys:
        if configuration1[key] != configuration2[key]:
            return False
    return True


def are_all_elements_equal(data_list):
    return data_list[1:] == data_list[:-1]


def get_min_configurations(configurations, number_of_configurations, comparison_key):
    """
    Get the configurations with minimum value according to the comparison key
    :param configurations: dictionary containing the configurations.
    :param number_of_configurations: number of configurations to return.
    :param comparison_key: name of the key used in the comparison.
    :return: a dictionary containing the best configurations.
    """
    tmp_configurations = copy.deepcopy(configurations)
    best_configurations = defaultdict(list)
    configurations_size = len(configurations[list(configurations.keys())[0]])

    # avoid requesting more configurations than possible
    number_of_configurations = min(number_of_configurations, configurations_size)

    for i in range(number_of_configurations):
        min_idx = np.argmin(tmp_configurations[comparison_key])
        for key in tmp_configurations:
            param_value = tmp_configurations[key][min_idx]
            best_configurations[key].append(param_value)
            del tmp_configurations[key][min_idx]
    return best_configurations


def get_min_feasible_configurations(
    configurations, number_of_configurations, comparison_key, feasible_parameter
):
    """
    Get the feasible configurations with minimum value according to the comparison key.
    If not enough feasible configurations are present, return the unfeasible configurations with minimum value.
    :param configurations: dictionary containing the configurations.
    :param number_of_configurations: number of configurations to return.
    :param comparison_key: name of the key used in the comparison.
    :param feasible_parameter: name of the key used to indicate feasibility.
    :return: a dictionary containing the best configurations.
    """
    feasible_configurations = {}
    unfeasible_configurations = {}

    configurations_size = len(configurations[list(configurations.keys())[0]])
    number_of_configurations = min(number_of_configurations, configurations_size)

    feasible_counter = 0
    for idx in range(configurations_size):
        configuration = get_single_configuration(configurations, idx)
        for key in configuration:
            configuration[key] = [configuration[key]]
        if configuration[feasible_parameter][0]:
            feasible_configurations = concatenate_data_dictionaries(
                feasible_configurations, configuration
            )
            feasible_counter += 1
        else:
            unfeasible_configurations = concatenate_data_dictionaries(
                unfeasible_configurations, configuration
            )

    if feasible_counter < number_of_configurations:
        missing_configurations = number_of_configurations - feasible_counter
        best_unfeasible_configurations = get_min_configurations(
            unfeasible_configurations, missing_configurations, comparison_key
        )
        best_configurations = concatenate_data_dictionaries(
            feasible_configurations, best_unfeasible_configurations
        )
    elif feasible_counter > number_of_configurations:
        best_configurations = get_min_configurations(
            feasible_configurations, number_of_configurations, comparison_key
        )
    else:
        best_configurations = feasible_configurations
    return best_configurations


####################################################
# Visualization
####################################################
def get_next_color():
    get_next_color.ccycle = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
        (0, 200, 0),
        (0, 0, 0),
        # get_next_color.ccycle = [(101, 153, 255), (0, 0, 0), (100, 100, 100), (150, 100, 150), (150, 150, 150),
        # (192, 192, 192), (255, 0, 0), (255, 153, 0), (199, 233, 180), (9, 112, 84),
        (0, 128, 0),
        (0, 0, 0),
        (199, 233, 180),
        (9, 112, 84),
        (170, 163, 57),
        (255, 251, 188),
        (230, 224, 123),
        (110, 104, 14),
        (49, 46, 0),
        (138, 162, 54),
        (234, 248, 183),
        (197, 220, 118),
        (84, 105, 14),
        (37, 47, 0),
        (122, 41, 106),
        (213, 157, 202),
        (165, 88, 150),
        (79, 10, 66),
        (35, 0, 29),
        (65, 182, 196),
        (34, 94, 168),
        (12, 44, 132),
        (79, 44, 115),
        (181, 156, 207),
        (122, 89, 156),
        (44, 15, 74),
        (18, 2, 33),
    ]
    get_next_color.color_count += 1
    if get_next_color.color_count > 33:
        return (0, 0, 0)
    else:
        (a, b, c) = get_next_color.ccycle[get_next_color.color_count - 1]
        return (float(a) / 255, float(b) / 255, float(c) / 255)


get_next_color.color_count = 0


####################################################
# Parameter file validation
####################################################
def extend_with_default(validator_class):
    """
    Initialize the json schema with the default values declared in the schema.json file.
    :param validator_class:
    :return:
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])
        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_class, {"properties": set_defaults})


def validate_json(parameters_file):
    """
    Validate a json file using Hypermapper's schema.
    :param paramters_file: json file to validate.
    :return: dictionary with the contents from the json file
    """
    filename, file_extension = os.path.splitext(parameters_file)
    assert file_extension == ".json"
    with open(parameters_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    DefaultValidatingDraft4Validator(schema).validate(config)

    return config


####################################################
# Parallel function computation
####################################################
def domain_decomposition_and_parallel_computation(*args):
    """
    Perform domain decomposition on the array "data_array" and then compute the partitions in parallel using the input function "function".
    The computation is done in parallel exploiting all usable processors (and hyperthreading).
    The number of processors used may be less than the actual number on embedded systems (to be debuged on these systems).
    - First argument "debug".
    - Second argument "function".
    - Third argument "concatenate_function.
    - Forth argument "data_array".
    - Fifth argument "number_of_cpus" is the number of cpus to use specified in the json file. If 0 it means to
    query the system from this function and do its best otherwise if forces the number of cpus to the number given.
    - Other arguments are passed to the function "function" in order.
    """
    assert len(args) >= 4
    debug = args[0]
    function = args[1]
    concatenate_function = args[2]
    data_array = args[3]
    number_of_cpus = args[4]
    function_arguments = [args[arg] for arg in range(len(args)) if arg >= 5]
    len_data_array = len(data_array)

    # mp.cpu_count() may cause problems, i.e. not detecting the right number of usable CPUs. In particular on embedded systems.
    # Implement the solution here if needed to fix this issue: https://stackoverflow.com/questions/31346974/portable-way-of-detecting-number-of-usable-cpus-in-python
    number_of_cpus_available = mp.cpu_count()
    if number_of_cpus == 0:
        number_of_cpus = number_of_cpus_available
    if (
        number_of_cpus > 8
    ):  # This is temporary, using a big number of CPUs may cause problems with current implementation
        number_of_cpus = 8
    chunks_multiplier = 1
    number_of_chunks = min(number_of_cpus * chunks_multiplier, len_data_array)
    print(
        "Number of available cpus:%d, number of cpus used:%d, number of chunks:%d, data array len: %d"
        % (number_of_cpus_available, number_of_cpus, number_of_chunks, len_data_array)
    )
    pool = mp.Pool(processes=number_of_cpus)
    float_number_of_chunks = float(number_of_chunks)
    data_array_parallel = {}
    # Domain decomposition
    for chunk in range(number_of_chunks - 1):
        data_array_parallel[chunk] = data_array[
            int(len_data_array * (chunk / float_number_of_chunks)) : int(
                len_data_array * ((chunk + 1) / float_number_of_chunks)
            )
        ]
    data_array_parallel[number_of_chunks - 1] = data_array[
        int(len_data_array * ((number_of_chunks - 1) / float_number_of_chunks)) :
    ]
    # data_array_1 = data_array[int(len_data_array * (0/4.)) : int(len_data_array * (1/4.))]
    # data_array_2 = data_array[int(len_data_array * (1/4.)) : int(len_data_array * (2/4.))]
    # data_array_3 = data_array[int(len_data_array * (2/4.)) : int(len_data_array * (3/4.))]
    # data_array_4 = data_array[int(len_data_array * (3/4.)): ]
    # debug = True
    if debug:
        print("In the domain_domain_decomposition_and_parallel_computation routine")
        for chunk in range(number_of_chunks):
            print("Chunk #%d, length=%d" % (chunk, len(data_array_parallel[chunk])))

    launch_parallel = [
        pool.apply_async(function, [data_array_parallel[chunk]] + function_arguments)
        for chunk in range(number_of_chunks)
    ]  # evaluate "function" asynchronously
    results_parallel = [
        launch_parallel[chunk].get(None) for chunk in range(number_of_chunks)
    ]
    # In the case the parallel computation has to be done on a function that doesn't need any additional parameter we can do this:
    # results = pool.map(function, [data_array_parallel[cpu] for cpu in range(number_of_cpus)])

    pool.close()
    pool.join()

    return concatenate_function(
        [results_parallel[cpu] for cpu in range(number_of_chunks)]
    )


####################################################
# Data conversion
####################################################
def data_tuples_to_dictionary(data_tuple, keys):
    new_dict = defaultdict(list)
    for entry in data_tuple:
        for idx, key in enumerate(keys):
            new_dict[key].append(entry[idx])
    return new_dict


def data_dictionary_to_tuple(data_dict, keys):
    tuples = []
    for idx in range(len(data_dict[keys[0]])):
        configuration = []
        for key in keys:
            configuration.append(data_dict[key][idx])
        tuples.append(tuple(configuration))
    return tuples


def data_tuples_to_dict_list(data_tuple, keys):
    dict_list = []
    for entry in data_tuple:
        configuration = {}
        for idx, key in enumerate(keys):
            configuration[key] = entry[idx]
        dict_list.append(configuration)
    return dict_list


def dict_list_to_matrix(dict_list, keys=None):
    matrix = []
    if keys == None:
        keys = dict_list[0].keys()
    for dictionary in dict_list:
        row = []
        for key in keys:
            row.append(dictionary[key])
        matrix.append(row)
    return matrix


def dict_of_lists_to_list_of_dicts(data_dict, keys=None):
    dict_list = []
    if keys == None:
        keys = list(data_dict.keys())
    for idx in range(len(data_dict[keys[0]])):
        configuration = {}
        for key in keys:
            configuration[key] = data_dict[key][idx]
        dict_list.append(configuration)
    return dict_list


def array_to_list_of_dicts(data_array, keys):
    return [
        {key: data_array[sample_nbr, dim] for dim, key in enumerate(keys)}
        for sample_nbr in range(len(data_array[:, 0]))
    ]


def dict_of_lists_to_numpy(input_dict, return_col_of_key):
    variable_names = list(input_dict.keys())
    new_array = np.zeros(
        (len(input_dict[variable_names[0]]), len(input_dict)), dtype=np.dtype(object)
    )

    # to get the column corresponding to a specific key
    col_of_key = {}
    for i, var in enumerate(variable_names):
        col_of_key[var] = i
        new_array[:, i] = input_dict[variable_names[i]]

    if return_col_of_key:
        return new_array, col_of_key
    return new_array


####################################################
# Data normalization
####################################################
def compute_std_and_max_point(data_array, selection_keys=None):
    """
    Compute the standard deviations and maxima points for a subset of keys of a dictionary.
    :param data_array: dictionary containing the data for the computations
    :param selection_keys: list containing the keys to use. The entire dictionary is used.
    :return: dictionary with the standard deviations and list with the maximum point for each objective
    """
    max_points = []
    standard_deviations = {}
    if selection_keys == None:
        selection_keys = list(data_array.keys())
    for key in selection_keys:
        X = np.array(data_array[key])

        standard_deviation = np.std(X, axis=0)
        standard_deviations[key] = standard_deviation

        max_points.append(max(data_array[key]))

    return standard_deviations, max_points


def normalize_with_std(data_array, standard_deviations, selection_keys=None):
    """
    Normalize a data dictionary using standard deviations.
    :param data_array: dictionary to be normalized
    :param standard_deviations: dictionary containing the standard deviation for each key
    :param selection_keys: list containing the keys to use. The entire dictionary is used.
    :return: normalized dictionary
    """
    if selection_keys == None:
        selection_keys = list(data_array.keys())
    for key in selection_keys:
        X = np.array(data_array[key])
        X /= standard_deviations[key]
        data_array[key] = X

    return data_array


####################################################
# Scalarization
####################################################
def reciprocate_weights(objective_weights):
    """
    Reciprocate weights so that they correlate when using modified_tchebyshev scalarization.
    :param objective_weights: a dictionary containing the weights for each objective.
    :return: a dictionary containing the reciprocated weights.
    """
    new_weights = {}
    total_weight = 0
    for objective in objective_weights:
        new_weights[objective] = 1 / objective_weights[objective]
        total_weight += new_weights[objective]

    for objective in new_weights:
        new_weights[objective] = new_weights[objective] / total_weight

    return new_weights


def compute_data_array_scalarization(
    data_array, objective_weights, objective_limits, scalarization_method
):
    """
    :param data_array: a dictionary containing the previously run points and their function values.
    :param objective_weights: a list containing the weights for each objective.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param scalarization_method: a string indicating which scalarization method to use.
    :return: a list of scalarized values for each point in data_array and the updated objective limits.
    """
    data_array_len = len(data_array[list(data_array.keys())[0]])
    tmp_objective_limits = copy.deepcopy(objective_limits)

    normalized_data_array = {}
    for objective in objective_limits:
        tmp_min = min(data_array[objective])
        tmp_objective_limits[objective][0] = min(
            tmp_min, tmp_objective_limits[objective][0]
        )
        tmp_max = max(data_array[objective])
        tmp_objective_limits[objective][1] = max(
            tmp_max, tmp_objective_limits[objective][1]
        )
        # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
        # We cannot optimize an objective when all values are the same, so we set it to 0
        if objective_limits[objective][1] == objective_limits[objective][0]:
            normalized_data_array[objective] = [0] * len(data_array[objective])
        else:
            normalized_data_array[objective] = [
                (x - tmp_objective_limits[objective][0])
                / (
                    tmp_objective_limits[objective][1]
                    - tmp_objective_limits[objective][0]
                )
                for x in data_array[objective]
            ]

    if scalarization_method == "linear":
        scalarized_objectives = np.zeros(data_array_len)
        for run_index in range(data_array_len):
            for objective in objective_weights:
                scalarized_objectives[run_index] += (
                    objective_weights[objective]
                    * normalized_data_array[objective][run_index]
                )
    # The paper does not propose this, we apply their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif scalarization_method == "tchebyshev":
        scalarized_objectives = np.zeros(data_array_len)
        for run_index in range(data_array_len):
            total_value = 0
            for objective in objective_weights:
                scalarized_value = objective_weights[objective] * abs(
                    normalized_data_array[objective][run_index]
                )
                scalarized_objectives[run_index] = max(
                    scalarized_value, scalarized_objectives[run_index]
                )
                total_value += scalarized_value
            scalarized_objectives[run_index] += 0.05 * total_value
    elif scalarization_method == "modified_tchebyshev":
        scalarized_objectives = np.full((data_array_len), float("inf"))
        reciprocated_weights = reciprocate_weights(objective_weights)
        for run_index in range(data_array_len):
            for objective in objective_weights:
                scalarized_value = reciprocated_weights[objective] * abs(
                    normalized_data_array[objective][run_index]
                )
                scalarized_objectives[run_index] = min(
                    scalarized_value, scalarized_objectives[run_index]
                )
            scalarized_objectives[run_index] = -scalarized_objectives[run_index]
    return scalarized_objectives, tmp_objective_limits


def sample_weight_bbox(
    optimization_metrics,
    objective_bounds,
    objective_limits,
    evaluations_per_optimization_iteration,
):
    """
    Sample lambdas for each objective following a uniform distribution with user-defined bounding boxes.
    If the user does not define bounding boxes, it defaults to [0, 1].
    :param optimization_metrics: a list containing the optimization objectives.
    :param objective_bounds: a dictionary containing the bounding boxes for each objective.
    :param evaluations_per_optimization_iteration: number of weight arrays to sample. Currently not used.
    :return: a dictionary containing the weight of each objective.
    """
    weight_list = []
    total_weight = 0.0
    for run_idx in range(evaluations_per_optimization_iteration):
        objective_weights = {}
        for objective in optimization_metrics:
            loc, scale = objective_bounds[objective]
            scale = (
                scale - loc
            )  # scipy.stats automatically does scale = scale + loc, we don't want that
            objective_weight = stats.uniform.rvs(loc=loc, scale=scale)
            objective_weights[objective] = objective_weight

            # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
            # We cannot optimize an objective when all values are the same, so we set its weight to 0.
            if objective_limits[objective][1] == objective_limits[objective][0]:
                objective_weights[objective] = 0
            else:
                objective_weights[objective] = (
                    objective_weights[objective] - objective_limits[objective][0]
                ) / (objective_limits[objective][1] - objective_limits[objective][0])
            total_weight += objective_weights[objective]
            if total_weight == 0:
                total_weight = 1

        for objective in objective_weights:
            objective_weights[objective] = objective_weights[objective] / total_weight
        weight_list.append(objective_weights)

    return weight_list


def sample_weight_flat(optimization_metrics, evaluations_per_optimization_iteration):
    """
    Sample lambdas for each objective following a dirichlet distribution with alphas equal to 1.
    In practice, this means we sample the weights uniformly from the set of possible weight vectors.
    :param optimization_metrics: a list containing the optimization objectives.
    :param evaluations_per_optimization_iteration: number of weight arrays to sample. Currently not used.
    :return: a dictionary containing the weight of each objective.
    """
    alphas = np.ones(len(optimization_metrics))
    sampled_weights = stats.dirichlet.rvs(
        alpha=alphas, size=evaluations_per_optimization_iteration
    )
    weight_list = []

    for run_idx in range(evaluations_per_optimization_iteration):
        objective_weights = {}
        for idx, objective in enumerate(optimization_metrics):
            objective_weights[objective] = sampled_weights[run_idx][idx]
        weight_list.append(objective_weights)

    return weight_list
