import sys
import os
import space
import numpy as np
import csv
import json
import copy
import datetime
from jsonschema import Draft4Validator, validators, exceptions
from utility_functions import *
from collections import defaultdict
from scipy import stats

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

def get_min_feasible_configurations(configurations, number_of_configurations, comparison_key, feasible_parameter):
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
    for idx in range(number_of_configurations):
        configuration = get_single_configuration(configurations, idx)
        for key in configuration:
            configuration[key] = [configuration[key]]
        if configurations[feasible_parameter]:
            feasible_configurations = concatenate_data_dictionaries(feasible_configurations, configuration)
            feasible_counter += 1
        else:
            unfeasible_configurations = concatenate_data_dictionaries(unfeasible_configurations, configuration)

    if feasible_counter < number_of_configurations:
        missing_configurations = number_of_configurations - feasible_counter
        best_unfeasible_configurations = get_min_configurations(unfeasible_configurations, missing_configurations, comparison_key)
        best_configurations = concatenate_data_dictionaries(feasible_configurations, best_unfeasible_configurations)
    elif feasible_counter > number_of_configurations:
        best_configurations = get_min_configurations(feasible_configurations, number_of_configurations, comparison_key)
    else:
        best_configurations = feasible_configurations
    return best_configurations

def get_neighbors(configuration, param_space):
    """
    Get the neighbors of a configuration, following the approach defined by SMAC
    http://www.cs.ubc.ca/labs/beta/Projects/SMAC/papers/11-LION5-SMAC.pdf
    :param configuration: dictionary containing the configuration we will generate neighbors for.
    :param param_space: a space object containing the search space.
    :return: a dictionary containing all the neighbors of 'configuration'.
    """
    neighbors = []
    input_parameters_objects = param_space.get_input_parameters_objects()
    input_parameters = param_space.get_input_parameters()
    numeric_neighbors = 4

    for param_name in input_parameters:
        param_type = param_space.get_type(param_name)
        param_object = input_parameters_objects[param_name]
        if (param_type == 'categorical') or ((param_type == 'ordinal') and len(param_object.get_discrete_values()) <= 5):
            for value in param_object.get_discrete_values():
                neighbor = []
                for input_param in input_parameters:
                    if input_param == param_name:
                        neighbor.append(value)
                    else:
                        neighbor.append(configuration[input_param])
                if neighbor not in neighbors:
                    neighbors.append(neighbor)
        elif (param_type == 'ordinal'):
            values = param_object.get_discrete_values()
            param_value = configuration[param_name]
            param_index = values.index(param_value)
            if param_index < numeric_neighbors/2:
                value_list = values[:numeric_neighbors+1]
            elif (len(values) - param_index) <= numeric_neighbors/2:
                value_list = values[-(numeric_neighbors+1):]
            else:
                lower_bound = int(param_index - numeric_neighbors/2)
                upper_bound = int(param_index + numeric_neighbors/2 + 1)
                value_list = values[lower_bound:upper_bound]
            for value in value_list:
                neighbor = []
                for input_param in input_parameters:
                    if input_param == param_name:
                        neighbor.append(value)
                    else:
                        neighbor.append(configuration[input_param])
                if neighbor not in neighbors:
                    neighbors.append(neighbor)
        elif (param_type == 'real') or (param_type == 'integer'):
            param_value = configuration[param_name]
            param_min = param_object.get_min()
            param_max = param_object.get_max()
            mean = (param_value - param_min)/(param_max - param_min)
            neighboring_values = stats.truncnorm.rvs(0, 1, loc=mean, scale=0.2, size=numeric_neighbors)
            neighboring_values = neighboring_values.tolist()
            neighboring_values.append(mean)
            for value in neighboring_values:
                neighbor = []
                if value < 0:
                    value = 0
                elif value > 1:
                    value = 1
                for input_param in input_parameters:
                    if input_param == param_name:
                        unscaled_value = param_object.from_range_0_1_to_parameter_value(value)
                        if type(unscaled_value) == list:
                            unscaled_value = unscaled_value[0]
                        neighbor.append(unscaled_value)
                    else:
                        neighbor.append(configuration[input_param])
                if neighbor not in neighbors:
                    neighbors.append(neighbor)
        else:
            print("Unsupported parameter type:", param_type)
            raise SystemExit
    return neighbors

def run_objective_function(
                        configurations,
                        hypermapper_mode,
                        param_space,
                        beginning_of_time,
                        run_directory,
                        local_search_data_array,
                        fast_addressing_of_data_array,
                        exhaustive_search_data_array,
                        exhaustive_search_fast_addressing_of_data_array,
                        scalarization_weights,
                        objective_limits,
                        scalarization_method,
                        enable_feasible_predictor=False,
                        evaluation_limit=float("inf"),
                        black_box_function=None,
                        number_of_cpus=0):
    """
    Evaluate a list of configurations using the black-box function being optimized.
    This method avoids evaluating repeated points by recovering their value from the history of evaluated points.
    :param configurations: list of configurations to evaluate.
    :param hypermapper_mode: which HyperMapper mode is being used.
    :param param_space: a space object containing the search space.
    :param beginning_of_time: timestamp of when the optimization started.
    :param run_directory: directory where HyperMapper is running.
    :param local_search_data_array: a dictionary containing all of the configurations that have been evaluated.
    :param fast_addressing_of_data_array: a dictionary containing evaluated configurations and their index in the local_search_data_array.
    :param exhaustive_search_data_array: dictionary containing all points and function values, used in exhaustive mode.
    :param exhaustive_search_fast_addressing_of_data_array: dictionary containing the index of each point in the exhaustive array.
    :param scalarization_weights: the weights used to scalarize the function value.
    :param objective_limits: dictionary containing the estimated min and max limits for each objective.
    :param scalarization_method: which method to use to scalarize multiple objectives.
    :param enable_feasible_predictor: whether to use constrained optimization.
    :param evaluation_limit: the maximum number of function evaluations allowed for the local search.
    :param black_box_function: the black_box_function being optimized in the local search.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: the best point found in the random sampling and local search.
    """
    new_configurations = []
    new_evaluations = {}
    previous_evaluations = defaultdict(list)
    number_of_new_evaluations = 0
    t0 = datetime.datetime.now()
    absolute_configuration_index = len(fast_addressing_of_data_array)
    function_values = {}

    for configuration in configurations:
        str_data = param_space.get_unique_hash_string_from_values(configuration)
        if str_data in fast_addressing_of_data_array:
            configuration_idx = fast_addressing_of_data_array[str_data]
            for key in local_search_data_array:
                previous_evaluations[key].append(local_search_data_array[key][configuration_idx])
        else:
            if absolute_configuration_index + number_of_new_evaluations < evaluation_limit:
                new_configurations.append(configuration)
                number_of_new_evaluations += 1

    t1 = datetime.datetime.now()
    if number_of_new_evaluations > 0:
        new_evaluations = param_space.run_configurations(
                                                        hypermapper_mode,
                                                        new_configurations,
                                                        beginning_of_time,
                                                        black_box_function,
                                                        exhaustive_search_data_array,
                                                        exhaustive_search_fast_addressing_of_data_array,
                                                        run_directory)

    all_evaluations = concatenate_data_dictionaries(previous_evaluations, new_evaluations)
    all_evaluations_size = len(all_evaluations[list(all_evaluations.keys())[0]])

    if enable_feasible_predictor:
        feasible_parameter = param_space.get_feasible_parameter()[0]
        feasibility_indicators = all_evaluations[feasible_parameter]
    else:
        # if no constraints, all points are feasible
        feasibility_indicators = [1]*all_evaluations_size

    scalarized_values, tmp_objective_limits = compute_data_array_scalarization(
                                                                            all_evaluations,
                                                                            scalarization_weights,
                                                                            objective_limits,
                                                                            None,
                                                                            scalarization_method)

    for objective in objective_limits:
        objective_limits[objective] = tmp_objective_limits[objective]

    for idx in range(number_of_new_evaluations):
        configuration = get_single_configuration(new_evaluations, idx)
        for key in configuration:
            local_search_data_array[key].append(configuration[key])

        str_data = param_space.get_unique_hash_string_from_values(configuration)
        fast_addressing_of_data_array[str_data] = absolute_configuration_index
        absolute_configuration_index += 1

    sys.stdout.write_to_logfile(("Time to run new configurations %10.4f sec\n" % ((datetime.datetime.now() - t1).total_seconds())))
    sys.stdout.write_to_logfile(("Total time to run configurations %10.4f sec\n" % ((datetime.datetime.now() - t0).total_seconds())))

    return list(scalarized_values), feasibility_indicators

def local_search(
                local_search_starting_points,
                local_search_random_points,
                param_space,
                fast_addressing_of_data_array,
                enable_feasible_predictor,
                optimization_function,
                optimization_function_parameters,
                scalarization_key,
                previous_points=None):
    """
    Optimize the acquisition function using a mix of random and local search.
    This algorithm random samples N points and then does a local search on the
    best points from the random search and the best points from previous iterations (if any).
    :param local_search_starting_points: an integer for the number of starting points for the local search. If 0, all points will be used.
    :param local_search_random_points: number of random points to sample before the local search.
    :param param_space: a space object containing the search space.
    :param fast_addressing_of_data_array: A list containing the points that were already explored.
    :param enable_feasible_predictor: whether to use constrained optimization.
    :param optimization_function: the function that will be optimized by the local search.
    :param optimization_function_parameters: a dictionary containing the parameters that will be passed to the optimization function.
    :param scalarization_key: the name given to the scalarized values.
    :param previous_points: previous points that have already been evaluated.
    :return: all points evaluted and the best point found by the local search.
    """
    t0 = datetime.datetime.now()
    tmp_fast_addressing_of_data_array = copy.deepcopy(fast_addressing_of_data_array)
    input_params = param_space.get_input_parameters()
    feasible_parameter = param_space.get_feasible_parameter()[0]
    data_array = {}
    end_of_search = False

    default_configuration = param_space.get_default_or_random_configuration()
    str_data = param_space.get_unique_hash_string_from_values(default_configuration)
    if str_data not in fast_addressing_of_data_array:
        tmp_fast_addressing_of_data_array[str_data] = 1
        if local_search_random_points - 1 > 0:
            configurations = [default_configuration] + param_space.random_sample_configurations_without_repetitions(tmp_fast_addressing_of_data_array, local_search_random_points-1)
    else:
        configurations = param_space.random_sample_configurations_without_repetitions(tmp_fast_addressing_of_data_array, local_search_random_points)

    # Passing the dictionary with ** expands the key-value pairs into function parameters
    function_values, feasibility_indicators = optimization_function(configurations=configurations, **optimization_function_parameters)

    # This will concatenate the entire neighbors array if all configurations were evaluated
    # but only the evaluated configurations if we reached the budget and did not evaluate all
    function_values_size = len(function_values)
    new_data_array = concatenate_list_of_dictionaries(configurations[:function_values_size])
    new_data_array[scalarization_key] = function_values
    if enable_feasible_predictor:
        new_data_array[feasible_parameter] = feasibility_indicators

    data_array = concatenate_data_dictionaries(data_array, new_data_array)

    # If some configurations were not evaluated, we reached the budget and must stop
    if function_values_size < len(configurations):
        sys.stdout.write_to_logfile("Out of budget, not all configurations were evaluated, stopping local search\n")
        end_of_search = True

    if enable_feasible_predictor:
        local_search_configurations = get_min_feasible_configurations(
                                                                    data_array,
                                                                    local_search_starting_points,
                                                                    scalarization_key,
                                                                    feasible_parameter)
    else:
        local_search_configurations = get_min_configurations(
                                                            data_array,
                                                            local_search_starting_points,
                                                            scalarization_key)

    if previous_points is not None:
        concatenation_keys = input_params + [scalarization_key]
        if enable_feasible_predictor:
            concatenation_keys + [feasible_parameter]
            best_previous = get_min_feasible_configurations(
                                                            previous_points,
                                                            local_search_starting_points,
                                                            scalarization_key,
                                                            feasible_parameter)
        else:
            best_previous = get_min_configurations(
                                                    previous_points,
                                                    local_search_starting_points,
                                                    scalarization_key)

        local_search_configurations = concatenate_data_dictionaries(local_search_configurations, best_previous, concatenation_keys)
        data_array = concatenate_data_dictionaries(data_array, previous_points, concatenation_keys)

    # best improvement local search
    search_iteration = 0
    while not end_of_search:
        new_local_search_configurations = {}
        end_of_search = True
        search_iteration += 1

        sys.stdout.write_to_logfile("Starting local search iteration: " + str(search_iteration) + "\n")
        iteration_t0 = datetime.datetime.now()
        for idx in range(len(local_search_configurations[list(local_search_configurations.keys())[0]])):
            configuration = get_single_configuration(local_search_configurations, idx)
            sys.stdout.write_to_logfile("Starting local search on configuration: " + str(configuration) + "\n")
            neighbors = get_neighbors(configuration, param_space)
            neighbors = data_tuples_to_dict_list(neighbors, input_params)

            function_values, feasibility_indicators  = optimization_function(configurations=neighbors, **optimization_function_parameters)

            function_values_size = len(function_values)
            new_data_array = concatenate_list_of_dictionaries(neighbors[:function_values_size])
            new_data_array[scalarization_key] = function_values
            if enable_feasible_predictor:
                new_data_array[feasible_parameter] = feasibility_indicators

            data_array = concatenate_data_dictionaries(data_array, new_data_array)

            # If some neighbors were not evaluated, we reached the budget and must stop
            if function_values_size < len(neighbors):
                sys.stdout.write_to_logfile("Out of budget, not all neighbors were evaluated, stopping local search\n")
                end_of_search = True
                break

            if enable_feasible_predictor:
                best_neighbor = get_min_feasible_configurations(new_data_array, 1, scalarization_key, feasible_parameter)
            else:
                best_neighbor = get_min_configurations(new_data_array, 1, scalarization_key)

            for key in configuration:
                configuration[key] = [configuration[key]]

            if are_configurations_equal(best_neighbor, configuration, input_params):
                sys.stdout.write_to_logfile("Local minimum found: " + str(best_neighbor) + '\n')
            else:
                sys.stdout.write_to_logfile("Replacing configuration by best neighbor: " + str(best_neighbor) + "\n")
                end_of_search = False
                new_local_search_configurations = concatenate_data_dictionaries(new_local_search_configurations, best_neighbor)
            sys.stdout.write_to_logfile(("Local search iteration time %10.4f sec\n" % ((datetime.datetime.now() - iteration_t0).total_seconds())))
        local_search_configurations = new_local_search_configurations

    # Compute best configuration found in the local search
    best_configuration = {}
    tmp_data_array = copy.deepcopy(data_array)
    best_configuration_idx = np.argmin(tmp_data_array[scalarization_key])
    for param in input_params:
        best_configuration[param] = tmp_data_array[param][best_configuration_idx]
    configuration_string = param_space.get_unique_hash_string_from_values(best_configuration)
    # If the best configuration has already been evaluated before, remove it and get the next best configuration
    while configuration_string in fast_addressing_of_data_array:
        for key in tmp_data_array:
            del tmp_data_array[key][best_configuration_idx]
        best_configuration_idx = np.argmin(tmp_data_array[scalarization_key])
        for param in input_params:
            best_configuration[param] = tmp_data_array[param][best_configuration_idx]
        configuration_string = param_space.get_unique_hash_string_from_values(best_configuration)

    sys.stdout.write_to_logfile(("Local search time %10.4f sec\n" % ((datetime.datetime.now() - t0).total_seconds())))

    return data_array, best_configuration


def main(config, black_box_function=None, output_file=""):
    """
    Run design-space exploration using random scalarizations.
    :param config: dictionary containing all the configuration parameters of this design-space exploration.
    :param output_file: a name for the file used to save the dse results.
    :return:
    """
    param_space = space.Space(config)

    run_directory = config["run_directory"]
    application_name = config["application_name"]
    hypermapper_mode = config["hypermapper_mode"]["mode"]

    if hypermapper_mode == "default":
        if black_box_function == None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)
    number_of_cpus = config["number_of_cpus"]
    local_search_random_points = config["local_search_random_points"]
    local_search_evaluation_limit = config["local_search_evaluation_limit"]
    if local_search_evaluation_limit == -1:
        local_search_evaluation_limit = float("inf")
    scalarization_key = config["scalarization_key"]
    scalarization_method = config["scalarization_method"]
    scalarization_weights = config["local_search_scalarization_weights"]
    if len(scalarization_weights) < len(optimization_metrics):
        print("Error: not enough scalarization weights. Received", len(scalarization_weights), "expected", len(optimization_metrics))
        raise SystemExit
    if sum(scalarization_weights) != 1:
        sys.stdout.write_to_logfile("Weights must sum 1. Normalizing weights.\n")
        for idx in range(len(scalarization_weights)):
            scalarization_weights[idx] = scalarization_weights[idx]/sum(scalarization_weights)
        sys.stdout.write_to_logfile("New weights:" + str(scalarization_weights) + "\n")
    objective_weights = {}
    objective_limits = {}
    for idx, objective in enumerate(optimization_metrics):
        objective_weights[objective] = scalarization_weights[idx]
        objective_limits[objective] = [float("inf"), float("-inf")]



    exhaustive_search_data_array = None
    exhaustive_search_fast_addressing_of_data_array = None
    if (hypermapper_mode == 'exhaustive'):
        exhaustive_file = config["hypermapper_mode"]["exhaustive_search_file"]
        print("Exhaustive mode, loading data from %s ..." % exhaustive_file)
        exhaustive_search_data_array, exhaustive_search_fast_addressing_of_data_array = param_space.load_data_file(exhaustive_file, debug=False, number_of_cpus=number_of_cpus)

    enable_feasible_predictor = False
    if "feasible_output" in config:
        feasible_output = config["feasible_output"]
        feasible_output_name = feasible_output["name"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
        enable_feasible_predictor_grid_search_on_recall_and_precision = feasible_output["enable_feasible_predictor_grid_search_on_recall_and_precision"]
        feasible_predictor_grid_search_validation_file = feasible_output["feasible_predictor_grid_search_validation_file"]
        feasible_parameter = param_space.get_feasible_parameter()

    local_search_starting_points = config["local_search_starting_points"]

    debug = False

    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if hypermapper_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    if output_file == "":
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
    else:
        output_data_file = output_file

    absolute_configuration_index = 0
    fast_addressing_of_data_array = {}
    local_search_fast_addressing_of_data_array = {}
    local_search_data_array = defaultdict(list)


    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = {}
    optimization_function_parameters['hypermapper_mode'] = hypermapper_mode
    optimization_function_parameters['param_space'] = param_space
    optimization_function_parameters['beginning_of_time'] = beginning_of_time
    optimization_function_parameters['run_directory'] = run_directory
    optimization_function_parameters['exhaustive_search_data_array'] = exhaustive_search_data_array
    optimization_function_parameters['exhaustive_search_fast_addressing_of_data_array'] = exhaustive_search_fast_addressing_of_data_array
    optimization_function_parameters['black_box_function'] = black_box_function
    optimization_function_parameters['number_of_cpus'] = number_of_cpus
    optimization_function_parameters['local_search_data_array'] = local_search_data_array
    optimization_function_parameters['fast_addressing_of_data_array'] = local_search_fast_addressing_of_data_array
    optimization_function_parameters['evaluation_limit'] = local_search_evaluation_limit
    optimization_function_parameters['scalarization_weights'] = objective_weights
    optimization_function_parameters['objective_limits'] = objective_limits
    optimization_function_parameters['scalarization_method'] = scalarization_method
    optimization_function_parameters['enable_feasible_predictor'] = enable_feasible_predictor

    print("Starting local search...")
    local_search_t0 = datetime.datetime.now()
    all_samples, best_configuration = local_search(
                                                local_search_starting_points,
                                                local_search_random_points,
                                                param_space,
                                                fast_addressing_of_data_array,
                                                enable_feasible_predictor,
                                                run_objective_function,
                                                optimization_function_parameters,
                                                scalarization_key
                                                )

    print("Local search finished after %d function evaluations"%(len(local_search_data_array[optimization_metrics[0]])))
    sys.stdout.write_to_logfile(("Local search time %10.4f sec\n" % ((datetime.datetime.now() - local_search_t0).total_seconds())))

    with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'w') as f:
        w = csv.writer(f)
        w.writerow(list(local_search_data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, local_search_data_array) for j in list(local_search_data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(local_search_data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])

    print("### End of the local search.")

if __name__ == "__main__":

    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the client-server mode we only want to log on the file.
    sys.stdout = Logger()

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
        raise SystemExit

    try:
        initial_directory = os.environ['PWD']
        hypermapper_home = os.environ['HYPERMAPPER_HOME']
        os.chdir(hypermapper_home)
    except:
        hypermapper_home = "."
        initial_directory = "."

    if not parameters_file.endswith('.json'):
        _, file_extension = os.path.splitext(parameters_file)
        print("Error: invalid file name. \nThe input file has to be a .json file not a %s" %file_extension)
        raise SystemExit
    with open(parameters_file, 'r') as f:
        config = json.load(f)

    json_schema_file = 'scripts/schema.json'
    with open(json_schema_file, 'r') as f:
        schema = json.load(f)

    try:
        DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    run_directory = config["run_directory"]
    if run_directory == ".":
        run_directory = initial_directory
        config["run_directory"] = run_directory
    log_file = config["log_file"]
    if log_file == "hypermapper_logfile.log":
        log_file = deal_with_relative_and_absolute_path(run_directory, log_file)
    sys.stdout = Logger(log_file)

    main(config)

    try:
        os.chdir(hypermapper_pwd)
    except:
        pass

    print("### End of the local search script.")
