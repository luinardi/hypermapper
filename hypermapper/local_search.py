import copy
import datetime
import os
import sys
import warnings
from collections import OrderedDict, defaultdict
from multiprocessing import Queue, cpu_count, Process, JoinableQueue

from scipy import stats
from threadpoolctl import threadpool_limits

import numpy as np

# ensure backward compatibility
try:
    from hypermapper import space
    from hypermapper.utility_functions import (
        concatenate_data_dictionaries,
        are_configurations_equal,
        get_single_configuration,
        data_tuples_to_dict_list,
        concatenate_list_of_dictionaries,
        compute_data_array_scalarization,
        deal_with_relative_and_absolute_path,
        dict_of_lists_to_list_of_dicts,
        array_to_list_of_dicts,
        dict_of_lists_to_numpy,
        get_min_configurations,
        get_min_feasible_configurations,
        get_output_data_file,
        create_output_data_file,
    )
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

    from hypermapper import space
    from hypermapper.utility_functions import (
        concatenate_data_dictionaries,
        are_configurations_equal,
        get_single_configuration,
        data_tuples_to_dict_list,
        concatenate_list_of_dictionaries,
        compute_data_array_scalarization,
        deal_with_relative_and_absolute_path,
        dict_of_lists_to_list_of_dicts,
        array_to_list_of_dicts,
        dict_of_lists_to_numpy,
        get_min_configurations,
        get_min_feasible_configurations,
        get_output_data_file,
        create_output_data_file,
    )


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
        if (param_type == "categorical") or (
            (param_type == "ordinal") and len(param_object.get_discrete_values()) <= 5
        ):
            for value in param_object.get_discrete_values():
                neighbor = []
                for input_param in input_parameters:
                    if input_param == param_name:
                        neighbor.append(value)
                    else:
                        neighbor.append(configuration[input_param])
                if neighbor not in neighbors:
                    neighbors.append(neighbor)
        elif param_type == "ordinal":
            values = param_object.get_discrete_values()
            param_value = configuration[param_name]
            param_index = values.index(param_value)
            if param_index < numeric_neighbors / 2:
                value_list = values[: numeric_neighbors + 1]
            elif (len(values) - param_index) <= numeric_neighbors / 2:
                value_list = values[-(numeric_neighbors + 1) :]
            else:
                lower_bound = int(param_index - numeric_neighbors / 2)
                upper_bound = int(param_index + numeric_neighbors / 2 + 1)
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
        elif (param_type == "real") or (param_type == "integer"):
            param_value = configuration[param_name]
            param_min = param_object.get_min()
            param_max = param_object.get_max()
            mean = (param_value - param_min) / (param_max - param_min)
            neighboring_values = stats.truncnorm.rvs(
                0, 1, loc=mean, scale=0.2, size=numeric_neighbors
            )
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
                        unscaled_value = param_object.from_range_0_1_to_parameter_value(
                            value
                        )
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
    output_data_file,
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
    number_of_cpus=0,
):
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
                previous_evaluations[key].append(
                    local_search_data_array[key][configuration_idx]
                )
        else:
            if (
                absolute_configuration_index + number_of_new_evaluations
                < evaluation_limit
            ):
                new_configurations.append(configuration)
                number_of_new_evaluations += 1

    t1 = datetime.datetime.now()
    if number_of_new_evaluations > 0:
        new_evaluations = param_space.run_configurations(
            hypermapper_mode,
            new_configurations,
            beginning_of_time,
            output_data_file,
            black_box_function,
            exhaustive_search_data_array,
            exhaustive_search_fast_addressing_of_data_array,
            run_directory,
        )

    all_evaluations = concatenate_data_dictionaries(
        previous_evaluations, new_evaluations
    )
    all_evaluations_size = len(all_evaluations[list(all_evaluations.keys())[0]])

    if enable_feasible_predictor:
        feasible_parameter = param_space.get_feasible_parameter()[0]
        feasibility_indicators = all_evaluations[feasible_parameter]
    else:
        # if no constraints, all points are feasible
        feasibility_indicators = [1] * all_evaluations_size

    scalarized_values, tmp_objective_limits = compute_data_array_scalarization(
        all_evaluations, scalarization_weights, objective_limits, scalarization_method
    )

    for objective in objective_limits:
        objective_limits[objective] = tmp_objective_limits[objective]

    for idx in range(number_of_new_evaluations):
        configuration = get_single_configuration(new_evaluations, idx)
        for key in configuration:
            local_search_data_array[key].append(configuration[key])

        str_data = param_space.get_unique_hash_string_from_values(configuration)
        fast_addressing_of_data_array[str_data] = absolute_configuration_index
        absolute_configuration_index += 1

    sys.stdout.write_to_logfile(
        (
            "Time to run new configurations %10.4f sec\n"
            % ((datetime.datetime.now() - t1).total_seconds())
        )
    )
    sys.stdout.write_to_logfile(
        (
            "Total time to run configurations %10.4f sec\n"
            % ((datetime.datetime.now() - t0).total_seconds())
        )
    )

    return list(scalarized_values), feasibility_indicators


def parallel_optimization_function(
    optimization_function_parameters,
    input_queue,
    output_queue,
    proc,
    optimization_function,
):
    temporary_parameters = optimization_function_parameters.copy()

    while True:
        input_parameters = input_queue.get()
        if input_parameters is None:
            input_queue.task_done()
            break

        configurations, split_index, conf_index = (
            input_parameters["partition"],
            input_parameters["split_index"],
            input_parameters["conf_index"],
        )
        temporary_parameters["configurations"] = configurations
        scalarized_values, feasibility_indicators = optimization_function(
            **temporary_parameters
        )
        output_queue.put(
            {
                "scalarized_values": scalarized_values,
                "feasibility_indicators": feasibility_indicators,
                "split_index": split_index,
                "conf_index": conf_index,
            }
        )
        input_queue.task_done()


def parallel_multistart_local_search(
    input_queue,
    output_queue,
    input_params,
    param_space,
    optimization_function_parameters,
    optimization_function,
    enable_feasible_predictor,
    scalarization_key,
    proc,
):
    feasible_parameter = param_space.get_feasible_parameter()[0]
    while True:
        config = input_queue.get()
        if config is None:
            input_queue.task_done()
            break
        iteration_data_array = {}
        configuration = config["config"]
        idx = config["idx"]
        logstring = (
            "Starting local search on configuration: " + str(configuration) + "\n"
        )
        while configuration is not None:
            neighbors = get_neighbors(configuration, param_space)
            neighbors = data_tuples_to_dict_list(neighbors, input_params)
            function_values, feasibility_indicators = optimization_function(
                configurations=neighbors, **optimization_function_parameters
            )
            function_values_size = len(function_values)
            new_data_array = concatenate_list_of_dictionaries(
                neighbors[:function_values_size]
            )
            new_data_array[scalarization_key] = function_values
            if enable_feasible_predictor:
                new_data_array[feasible_parameter] = feasibility_indicators
            iteration_data_array = concatenate_data_dictionaries(
                iteration_data_array, new_data_array
            )
            """
            The out of budget stopping criterion is removed for now, since it's not deemed relevant for this algorithm!
            """
            if enable_feasible_predictor:
                best_neighbor = get_min_feasible_configurations(
                    new_data_array, 1, scalarization_key, feasible_parameter
                )
            else:
                best_neighbor = get_min_configurations(
                    new_data_array, 1, scalarization_key
                )

            for key in configuration:
                configuration[key] = [configuration[key]]
            if are_configurations_equal(best_neighbor, configuration, input_params):
                logstring += "Local minimum found: " + str(best_neighbor) + "\n"
                output_queue.put(
                    {"data_array": iteration_data_array, "logstring": logstring}
                )
                configuration = None
                input_queue.task_done()
            else:
                logstring += (
                    "Replacing configuration by best neighbor: "
                    + str(best_neighbor)
                    + "\n"
                )
                configuration = {key: value[0] for key, value in best_neighbor.items()}


def local_search(
    local_search_starting_points,
    local_search_random_points,
    param_space,
    fast_addressing_of_data_array,
    enable_feasible_predictor,
    optimization_function,
    optimization_function_parameters,
    scalarization_key,
    number_of_cpus,
    previous_points=None,
    profiling=None,
    noise=False,
):
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
    if number_of_cpus == 0:
        number_of_cpus = cpu_count()
    t0 = datetime.datetime.now()
    tmp_fast_addressing_of_data_array = copy.deepcopy(fast_addressing_of_data_array)
    input_params = param_space.get_input_parameters()
    feasible_parameter = param_space.get_feasible_parameter()[0]
    data_array = {}
    end_of_search = False
    # percentage of oversampling for the local search starting points
    oversampling_factor = 2

    default_configuration = param_space.get_default_or_random_configuration()
    str_data = param_space.get_unique_hash_string_from_values(default_configuration)
    if str_data not in fast_addressing_of_data_array:
        tmp_fast_addressing_of_data_array[str_data] = 1
    if param_space.get_space_size() < local_search_random_points:
        all_configurations = dict_of_lists_to_list_of_dicts(param_space.get_space())
        half_of_points = int(len(all_configurations) / 2)
        uniform_configurations = all_configurations[0:half_of_points]
        prior_configurations = all_configurations[half_of_points::]
    else:
        uniform_configurations = param_space.get_random_configuration(
            size=local_search_random_points, use_priors=False, return_as_array=True
        )
        prior_configurations = param_space.get_random_configuration(
            size=local_search_random_points, use_priors=True, return_as_array=True
        )

        uniform_configurations = array_to_list_of_dicts(
            uniform_configurations, param_space.get_input_parameters()
        )
        prior_configurations = array_to_list_of_dicts(
            prior_configurations, param_space.get_input_parameters()
        )

    sampling_time = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        ("Total RS time %10.4f sec\n" % ((sampling_time - t0).total_seconds()))
    )

    # check that the number of configurations are not less than the number of CPUs
    min_number_of_configs = min(len(uniform_configurations), len(prior_configurations))
    if min_number_of_configs < number_of_cpus:
        number_of_cpus = min_number_of_configs

    # to avoid having partitions with no samples, it's necessary to compute a floor for the number of partitions for small sample spaces
    # alternatively, an arbitraty number of samples could be set for the number of points where we do not have to partition (since it will be quick anyway)
    min_number_per_partition = min_number_of_configs / number_of_cpus
    partitions_per_cpu = min(10, int(min_number_per_partition))
    if number_of_cpus == 1:
        function_values_uniform, feasibility_indicators_uniform = optimization_function(
            configurations=uniform_configurations, **optimization_function_parameters
        )
        function_values_prior, feasibility_indicators_prior = optimization_function(
            configurations=prior_configurations, **optimization_function_parameters
        )
    else:
        # the number of splits of the list of input points that each process is expected to handle

        uniform_partition_fraction = len(uniform_configurations) / (
            partitions_per_cpu * number_of_cpus
        )
        prior_partition_fraction = len(prior_configurations) / (
            partitions_per_cpu * number_of_cpus
        )
        uniform_partition = [
            uniform_configurations[
                int(i * uniform_partition_fraction) : int(
                    (i + 1) * uniform_partition_fraction
                )
            ]
            for i in range(partitions_per_cpu * number_of_cpus)
        ]
        prior_partition = [
            prior_configurations[
                int(i * prior_partition_fraction) : int(
                    (i + 1) * prior_partition_fraction
                )
            ]
            for i in range(partitions_per_cpu * number_of_cpus)
        ]

        # Define a process queue and the processes, each containing half uniform and half prior partitions
        # as arguments to the nested function along with the queue
        input_queue = JoinableQueue()
        for i in range(number_of_cpus * partitions_per_cpu):
            combined_partition = uniform_partition[i] + prior_partition[i]
            input_queue.put(
                {
                    "partition": combined_partition,
                    "split_index": len(uniform_partition[i]),
                    "conf_index": i,
                }
            )
        output_queue = Queue()

        processes = [
            Process(
                target=parallel_optimization_function,
                args=(
                    optimization_function_parameters,
                    input_queue,
                    output_queue,
                    i,
                    optimization_function,
                ),
            )
            for i in range(number_of_cpus)
        ]

        function_values_uniform, feasibility_indicators_uniform = [{}] * len(
            uniform_configurations
        ), [{}] * len(uniform_configurations)
        function_values_prior, feasibility_indicators_prior = [{}] * len(
            prior_configurations
        ), [{}] * len(prior_configurations)

        # starting the processes and ensuring there's nothing more to process - joining the input queue when it's empty
        with threadpool_limits(limits=1):
            for process in processes:
                process.start()
                input_queue.put(None)
            input_queue.join()

        # the index on which to split the output
        for i in range(number_of_cpus * partitions_per_cpu):
            # would like this queue call to be non-blocking, but that does not work since the processes would need to be closed (?) for that to reliably work
            result = output_queue.get()
            scalarized_values, feasibility_indicators, split_index, conf_index = (
                result["scalarized_values"],
                result["feasibility_indicators"],
                result["split_index"],
                result["conf_index"],
            )
            # since half of the result is uniform and half is prior, it needs splitting in the middle of the resulting arrays
            function_values_uniform[
                int(conf_index * uniform_partition_fraction) : int(
                    conf_index * uniform_partition_fraction
                )
                + split_index
            ] = scalarized_values[0:split_index]
            feasibility_indicators_uniform[
                int(conf_index * uniform_partition_fraction) : int(
                    conf_index * uniform_partition_fraction
                )
                + split_index
            ] = feasibility_indicators[0:split_index]
            function_values_prior[
                int(conf_index * prior_partition_fraction) : int(
                    conf_index * prior_partition_fraction
                )
                + (len(scalarized_values) - split_index)
            ] = scalarized_values[split_index::]
            feasibility_indicators_prior[
                int(conf_index * prior_partition_fraction) : int(
                    conf_index * prior_partition_fraction
                )
                + (len(scalarized_values) - split_index)
            ] = feasibility_indicators[split_index::]

        # Safeguard so ensure the processes actually stop - ensures no process waits for more input and quits the MSLS function

        input_queue.close()
        output_queue.close()
        for i in range(len(processes)):
            processes[i].join()

    acquisition_time = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        (
            "Optimization function time %10.4f sec\n"
            % (acquisition_time - sampling_time).total_seconds()
        )
    )

    # This will concatenate the entire neighbors array if all configurations were evaluated
    # but only the evaluated configurations if we reached the budget and did not evaluate all
    function_values_uniform_size = len(function_values_uniform)
    new_data_array_uniform = concatenate_list_of_dictionaries(
        uniform_configurations[:function_values_uniform_size]
    )
    new_data_array_uniform[scalarization_key] = function_values_uniform

    function_values_prior_size = len(function_values_prior)
    new_data_array_prior = concatenate_list_of_dictionaries(
        prior_configurations[:function_values_prior_size]
    )
    new_data_array_prior[scalarization_key] = function_values_prior

    if enable_feasible_predictor:
        new_data_array_uniform[feasible_parameter] = feasibility_indicators_uniform
        new_data_array_prior[feasible_parameter] = feasibility_indicators_prior

    new_data_array = concatenate_data_dictionaries(
        new_data_array_uniform, new_data_array_prior
    )
    data_array = concatenate_data_dictionaries(data_array, new_data_array)

    # If some configurations were not evaluated, we reached the budget and must stop
    if (function_values_uniform_size < len(uniform_configurations)) or (
        function_values_prior_size < len(prior_configurations)
    ):
        sys.stdout.write_to_logfile(
            "Out of budget, not all configurations were evaluated, stopping local search\n"
        )
        end_of_search = True

    best_nbr_of_points = local_search_starting_points * oversampling_factor
    if enable_feasible_predictor:
        local_search_configurations_uniform = get_min_feasible_configurations(
            new_data_array_uniform,
            best_nbr_of_points,
            scalarization_key,
            feasible_parameter,
        )
        local_search_configurations_prior = get_min_feasible_configurations(
            new_data_array_prior,
            best_nbr_of_points,
            scalarization_key,
            feasible_parameter,
        )
    else:
        local_search_configurations_uniform = get_min_configurations(
            new_data_array_uniform, best_nbr_of_points, scalarization_key
        )
        local_search_configurations_prior = get_min_configurations(
            new_data_array_prior, best_nbr_of_points, scalarization_key
        )

    local_search_configurations = concatenate_data_dictionaries(
        local_search_configurations_uniform, local_search_configurations_prior
    )

    if previous_points is not None:
        concatenation_keys = input_params + [scalarization_key]
        if enable_feasible_predictor:
            concatenation_keys + [feasible_parameter]
            best_previous = get_min_feasible_configurations(
                previous_points,
                local_search_starting_points,
                scalarization_key,
                feasible_parameter,
            )
        else:
            best_previous = get_min_configurations(
                previous_points, local_search_starting_points, scalarization_key
            )

        local_search_configurations = concatenate_data_dictionaries(
            local_search_configurations, best_previous, concatenation_keys
        )
        data_array = concatenate_data_dictionaries(
            data_array, previous_points, concatenation_keys
        )

    local_search_points_numpy, col_of_keys = dict_of_lists_to_numpy(
        local_search_configurations, return_col_of_key=True
    )
    uniform_points = local_search_points_numpy[0:best_nbr_of_points]
    prior_points = local_search_points_numpy[
        best_nbr_of_points : best_nbr_of_points * 2
    ]
    best_previous_points = local_search_points_numpy[best_nbr_of_points * 2 : :]

    (
        best_previous_points,
        prior_points,
        uniform_points,
    ) = param_space.remove_duplicate_configs(
        best_previous_points,
        prior_points,
        uniform_points,
        ignore_columns=col_of_keys["scalarization"],
    )
    combined_unique_points = np.concatenate(
        (
            uniform_points[0:local_search_starting_points],
            prior_points[0:local_search_starting_points],
            best_previous_points[0:local_search_starting_points],
        ),
        axis=0,
    )
    local_search_configurations = {
        key: combined_unique_points[:, column].tolist()
        for key, column in col_of_keys.items()
    }

    data_collection_time = datetime.datetime.now()
    number_of_configurations = len(
        local_search_configurations[list(local_search_configurations.keys())[0]]
    )
    sys.stdout.write_to_logfile(
        "Starting local search iteration: "
        + ", #configs:"
        + str(number_of_configurations)
        + "\n"
    )
    input_queue = JoinableQueue()
    output_queue = Queue()
    # puts each configuration in a queue to be evaluated in parallel
    for idx in range(number_of_configurations):
        input_queue.put(
            {
                "config": get_single_configuration(local_search_configurations, idx),
                "idx": idx,
            }
        )
        sys.stdout.write_to_logfile((f"{idx}, \n"))

    for i in range(number_of_cpus):
        input_queue.put(None)

    if number_of_cpus == 1:
        parallel_multistart_local_search(
            input_queue,
            output_queue,
            input_params,
            param_space,
            optimization_function_parameters,
            optimization_function,
            enable_feasible_predictor,
            scalarization_key,
            0,
        )
        input_queue.join()

    else:
        processes = [
            Process(
                target=parallel_multistart_local_search,
                args=(
                    input_queue,
                    output_queue,
                    input_params,
                    param_space,
                    optimization_function_parameters,
                    optimization_function,
                    enable_feasible_predictor,
                    scalarization_key,
                    i,
                ),
            )
            for i in range(number_of_cpus)
        ]

        with threadpool_limits(limits=1):
            for process in processes:
                process.start()
            input_queue.join()

    result_array = {}
    for i in range(number_of_configurations):
        result = output_queue.get()
        sys.stdout.write_to_logfile(result["logstring"], msg_is_verbose=True)
        result_array = concatenate_data_dictionaries(result_array, result["data_array"])
    data_array = concatenate_data_dictionaries(result_array, data_array)

    input_queue.close()
    output_queue.close()

    if number_of_cpus != 1:
        for i in range(len(processes)):
            processes[i].join()

    local_search_time = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        (
            "Multi-start LS time %10.4f sec\n"
            % (local_search_time - acquisition_time).total_seconds()
        )
    )
    # Compute best configuration found in the local search
    best_configuration = {}
    tmp_data_array = copy.deepcopy(data_array)
    best_configuration_idx = np.argmin(tmp_data_array[scalarization_key])
    for param in input_params:
        best_configuration[param] = tmp_data_array[param][best_configuration_idx]
    configuration_string = param_space.get_unique_hash_string_from_values(
        best_configuration
    )
    # If the best configuration has already been evaluated before, remove it and get the next best configuration
    while configuration_string in fast_addressing_of_data_array:
        for key in tmp_data_array:
            del tmp_data_array[key][best_configuration_idx]
        best_configuration_idx = np.argmin(tmp_data_array[scalarization_key])
        for param in input_params:
            best_configuration[param] = tmp_data_array[param][best_configuration_idx]
        configuration_string = param_space.get_unique_hash_string_from_values(
            best_configuration
        )

    post_MSLS_time = datetime.datetime.now()

    sys.stdout.write_to_logfile(
        ("MSLS time %10.4f sec\n" % (post_MSLS_time - acquisition_time).total_seconds())
    )
    if profiling is not None:
        profiling.add("(LS) Random sampling time", (sampling_time - t0).total_seconds())
        profiling.add(
            "(LS) Acquisition evaluation time",
            (acquisition_time - sampling_time).total_seconds(),
        )
        profiling.add(
            "(LS) Data collection time",
            (data_collection_time - acquisition_time).total_seconds(),
        )
        profiling.add(
            "(LS) Multi-start LS time",
            (local_search_time - data_collection_time).total_seconds(),
        )
        profiling.add(
            "(LS) Post-MSLS data treatment time",
            (post_MSLS_time - local_search_time).total_seconds(),
        )

    return data_array, best_configuration


def main(config, black_box_function=None, profiling=None):
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

    noise = config["noise"]
    output_data_file = get_output_data_file(
        config["output_data_file"], run_directory, application_name
    )
    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)
    # local search will not produce reasonable output if run in parallel - it is therefore disabled
    number_of_cpus = 1
    local_search_random_points = config["local_search_random_points"]
    local_search_evaluation_limit = config["local_search_evaluation_limit"]
    if local_search_evaluation_limit == -1:
        local_search_evaluation_limit = float("inf")
    scalarization_key = config["scalarization_key"]
    scalarization_method = config["scalarization_method"]
    scalarization_weights = config["local_search_scalarization_weights"]
    if len(scalarization_weights) < len(optimization_metrics):
        print(
            "Error: not enough scalarization weights. Received",
            len(scalarization_weights),
            "expected",
            len(optimization_metrics),
        )
        raise SystemExit
    if sum(scalarization_weights) != 1:
        sys.stdout.write_to_logfile("Weights must sum 1. Normalizing weights.\n")
        for idx in range(len(scalarization_weights)):
            scalarization_weights[idx] = scalarization_weights[idx] / sum(
                scalarization_weights
            )
        sys.stdout.write_to_logfile("New weights:" + str(scalarization_weights) + "\n")
    objective_weights = {}
    objective_limits = {}
    for idx, objective in enumerate(optimization_metrics):
        objective_weights[objective] = scalarization_weights[idx]
        objective_limits[objective] = [float("inf"), float("-inf")]

    exhaustive_search_data_array = None
    exhaustive_search_fast_addressing_of_data_array = None
    if hypermapper_mode == "exhaustive":
        exhaustive_file = config["hypermapper_mode"]["exhaustive_search_file"]
        print("Exhaustive mode, loading data from %s ..." % exhaustive_file)
        (
            exhaustive_search_data_array,
            exhaustive_search_fast_addressing_of_data_array,
        ) = param_space.load_data_file(
            exhaustive_file, debug=False, number_of_cpus=number_of_cpus
        )

    enable_feasible_predictor = False
    if "feasible_output" in config:
        feasible_output = config["feasible_output"]
        feasible_output_name = feasible_output["name"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
        enable_feasible_predictor_grid_search_on_recall_and_precision = feasible_output[
            "enable_feasible_predictor_grid_search_on_recall_and_precision"
        ]
        feasible_predictor_grid_search_validation_file = feasible_output[
            "feasible_predictor_grid_search_validation_file"
        ]
        feasible_parameter = param_space.get_feasible_parameter()

    local_search_starting_points = config["local_search_starting_points"]

    debug = False

    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    sys.stdout.set_verbose_mode(config["verbose_logging"])
    if hypermapper_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    absolute_configuration_index = 0
    fast_addressing_of_data_array = {}
    local_search_fast_addressing_of_data_array = {}
    local_search_data_array = defaultdict(list)

    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = {}
    optimization_function_parameters["hypermapper_mode"] = hypermapper_mode
    optimization_function_parameters["param_space"] = param_space
    optimization_function_parameters["beginning_of_time"] = beginning_of_time
    optimization_function_parameters["run_directory"] = run_directory
    optimization_function_parameters["output_data_file"] = output_data_file
    optimization_function_parameters[
        "exhaustive_search_data_array"
    ] = exhaustive_search_data_array
    optimization_function_parameters[
        "exhaustive_search_fast_addressing_of_data_array"
    ] = exhaustive_search_fast_addressing_of_data_array
    optimization_function_parameters["black_box_function"] = black_box_function
    optimization_function_parameters["number_of_cpus"] = number_of_cpus
    optimization_function_parameters[
        "local_search_data_array"
    ] = local_search_data_array
    optimization_function_parameters[
        "fast_addressing_of_data_array"
    ] = local_search_fast_addressing_of_data_array
    optimization_function_parameters["evaluation_limit"] = local_search_evaluation_limit
    optimization_function_parameters["scalarization_weights"] = objective_weights
    optimization_function_parameters["objective_limits"] = objective_limits
    optimization_function_parameters["scalarization_method"] = scalarization_method
    optimization_function_parameters[
        "enable_feasible_predictor"
    ] = enable_feasible_predictor

    create_output_data_file(
        output_data_file, param_space.get_input_output_and_timestamp_parameters()
    )

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
        scalarization_key,
        number_of_cpus,
        profiling=profiling,
        noise=noise,
    )

    print(
        "Local search finished after %d function evaluations"
        % (len(local_search_data_array[optimization_metrics[0]]))
    )

    print("### End of the local search.")
    return local_search_data_array
