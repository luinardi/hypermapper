###############################################################################################################################
# This script implements a simplification of the evolutionary process proposed by Real et al.: https://arxiv.org/abs/1802.01548v7#
###############################################################################################################################
import copy
import datetime
import os
import sys
import warnings
from collections import OrderedDict, defaultdict

import numpy.random as rd
from jsonschema import exceptions

# ensure backward compatibility
try:
    from hypermapper import space
    from hypermapper.utility_functions import (
        concatenate_data_dictionaries,
        get_single_configuration,
        concatenate_list_of_dictionaries,
        deal_with_relative_and_absolute_path,
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
        get_single_configuration,
        concatenate_list_of_dictionaries,
        deal_with_relative_and_absolute_path,
        get_output_data_file,
        create_output_data_file,
    )


def mutation(param_space, config, mutation_rate, list=False):
    """
    Mutates given configuration.
    :param param_space: space.Space(), will give us information about parameters
    :param configs: list of configurations.
    :param mutation_rate: integer for how many parameters to mutate
    :param list: boolean whether returning one or more alternative configs
    :return: list of dicts, list of mutated configurations
    """

    parameter_object_list = param_space.get_input_parameters_objects()
    rd_config = dict()
    for name, obj in parameter_object_list.items():
        x = obj.randomly_select()
        single_valued_param = False
        param_type = param_space.get_type(name)

        if param_type == "real" or param_type == "integer":
            if obj.get_max() == obj.get_min():
                single_valued_param = True
        else:
            if obj.get_size() == 1:
                single_valued_param = True
        mutation_attempts = 0
        while x == config[name] and single_valued_param == False:
            x = obj.randomly_select()
            mutation_attempts += 1
            if mutation_attempts > 1000000:
                break
        rd_config[name] = x
    parameter_names_list = param_space.get_input_parameters()
    nbr_params = len(parameter_names_list)

    configs = []
    n_configs = nbr_params if list else 1

    for _ in range(n_configs):
        indices = rd.permutation(nbr_params)[:mutation_rate]
        for idx in indices:
            mutation_param = parameter_names_list[idx]
            # Should I do something if they are the same?
            temp = config.copy()
            temp[mutation_param] = rd_config[mutation_param]
            configs.append(temp)

    return configs


# Taken from local_search and slightly modified
def run_objective_function(
    configurations,
    hypermapper_mode,
    param_space,
    beginning_of_time,
    output_data_file,
    run_directory,
    evolution_data_array,
    fast_addressing_of_data_array,
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
    hypermapper_mode == "default"
    :param param_space: a space object containing the search space.
    :param beginning_of_time: timestamp of when the optimization started.
    :param run_directory: directory where HyperMapper is running.
    :param evolution_data_array: a dictionary containing all of the configurations that have been evaluated.
    :param fast_addressing_of_data_array: a dictionary containing evaluated configurations and their index in
    the evolution_data_array.
    :param enable_feasible_predictor: whether to use constrained optimization.
    :param evaluation_limit: the maximum number of function evaluations allowed for the evolutionary search.
    :param black_box_function: the black_box_function being optimized in the evolutionary search.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: configurations with evaluations for all points in configurations and the number of evaluated configurations
    """
    new_configurations = []
    new_evaluations = {}
    previous_evaluations = defaultdict(list)
    number_of_new_evaluations = 0
    t0 = datetime.datetime.now()
    absolute_configuration_index = len(fast_addressing_of_data_array)

    # Adds configutations to new if they have not been evaluated before
    for configuration in configurations:
        str_data = param_space.get_unique_hash_string_from_values(configuration)
        if str_data in fast_addressing_of_data_array:
            configuration_idx = fast_addressing_of_data_array[str_data]
            for key in evolution_data_array:
                previous_evaluations[key].append(
                    evolution_data_array[key][configuration_idx]
                )
        else:
            if (
                absolute_configuration_index + number_of_new_evaluations
                < evaluation_limit
            ):
                new_configurations.append(configuration)
                number_of_new_evaluations += 1

    # Evaluates new configurations. If there is any
    t1 = datetime.datetime.now()
    if number_of_new_evaluations > 0:
        new_evaluations = param_space.run_configurations(
            hypermapper_mode,
            new_configurations,
            beginning_of_time,
            output_data_file,
            black_box_function,
            run_directory=run_directory,
        )

    # Values for all given configurations
    all_evaluations = concatenate_data_dictionaries(
        previous_evaluations, new_evaluations
    )
    all_evaluations_size = len(all_evaluations[list(all_evaluations.keys())[0]])

    population = list()
    for idx in range(number_of_new_evaluations):
        configuration = get_single_configuration(new_evaluations, idx)
        population.append(configuration)
        for key in configuration:
            evolution_data_array[key].append(configuration[key])

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

    return population, all_evaluations_size


def evolution(
    population_size,
    generations,
    mutation_rate,
    crossover,
    regularize,
    batch_size,
    fitness_measure,
    param_space,
    fast_addressing_of_data_array,
    optimization_function,
    optimization_function_parameters,
    profiling=None,
):
    """
    Do the entire evolutinary process from config to best config
    :param population_size: an integer for the number of configs to keep. All will be initiated randomly
    :param generations: an integer for the number of iterations through the evolutionary loop
    :param mutation_rate: an integer for the number of parameters to change in a mutation
    :param crossover: a boolean whether to use crossover in the algorithm
    :param regularize: boolean, whether to use regularized or non-regularized evolution strategy
    :param batch_size: an integer for how many individuals to compare in a generation
    :param fitness_measure: a string name of the objective that should be optimized
    :param param_space: a space object containing the search space.
    :param fast_addressing_of_data_array: an array that keeps track of all evaluated configurations
    :param optimization_function: the function that will be optimized by the evolutionary search.
    :param optimization_function_parameters: a dictionary containing the parameters that will be passed to the
    optimization function.
    :return: all points evaluted and the best config at each generation of the Evolutionary Algorithm.
    """

    t0 = datetime.datetime.now()
    tmp_fast_addressing_of_data_array = copy.deepcopy(fast_addressing_of_data_array)
    input_params = param_space.get_input_parameters()
    data_array = {}

    ### Initialize a random population ###
    default_configuration = param_space.get_default_or_random_configuration()
    str_data = param_space.get_unique_hash_string_from_values(default_configuration)
    if str_data not in fast_addressing_of_data_array:
        tmp_fast_addressing_of_data_array[str_data] = 1
        if population_size - 1 > 0:  # Will always be true
            configurations = [
                default_configuration
            ] + param_space.random_sample_configurations_without_repetitions(
                tmp_fast_addressing_of_data_array, population_size - 1
            )
    else:
        configurations = param_space.random_sample_configurations_without_repetitions(
            tmp_fast_addressing_of_data_array, population_size
        )

    population, function_values_size = optimization_function(
        configurations=configurations, **optimization_function_parameters
    )

    # This will concatenate the entire data array if all configurations were evaluated
    new_data_array = concatenate_list_of_dictionaries(
        configurations[:function_values_size]
    )
    data_array = concatenate_data_dictionaries(data_array, new_data_array)

    ### Evolutionary loop ###
    for gen in range(1, generations + 1):
        if not gen % 10:
            print("Now we are at generation: ", gen)

        # pick a random batch from the population and find the two best and the worst of the batch
        cand_idxs = rd.permutation(len(population))[:batch_size]
        infty = float("inf")
        best = (-1, infty)
        second = (-1, infty)
        worst = (-1, -infty)
        for ci in cand_idxs:
            val = population[ci][fitness_measure]
            if val < best[1]:
                second = best
                best = (ci, val)
            elif val < second[1]:
                second = (ci, val)
            if val > worst[1]:
                worst = (ci, val)

        # checks that candidate loop was successful
        if min(best[0], second[0], worst[0]) < 0:
            print("failed to fined best and/or worst individual. Script will terminate")
            sys.exit()

        # Make a child by copy/crossover from parent(s)
        child = dict()
        parent = population[best[0]]
        if crossover:
            parent2 = population[second[0]]
            for param in input_params:
                if rd.uniform(0, 1) < 0.5:
                    child[param] = parent[param]
                else:
                    child[param] = parent2[param]
        else:
            for param in input_params:
                child[param] = parent[param]

        # Get mutation candidates, evaluate and add to population
        child_list = mutation(param_space, child, mutation_rate, list=True)
        need_random = True
        for c in child_list:
            evaluated_child_list, func_val_size = optimization_function(
                configurations=[c], **optimization_function_parameters
            )

            if evaluated_child_list:
                new_data_array = concatenate_list_of_dictionaries([c][:func_val_size])
                data_array = concatenate_data_dictionaries(data_array, new_data_array)

                population.append(evaluated_child_list[0])
                need_random = False
                break

        # If no new configs where found, draw some random configurations instead.
        if need_random:
            tmp_fast_addressing_of_data_array = copy.deepcopy(
                optimization_function_parameters["fast_addressing_of_data_array"]
            )

            random_children = (
                param_space.random_sample_configurations_without_repetitions(
                    tmp_fast_addressing_of_data_array, 1
                )
            )

            evaluated_random_children, func_val_size = optimization_function(
                configurations=random_children, **optimization_function_parameters
            )
            new_data_array = concatenate_list_of_dictionaries(
                random_children[:func_val_size]
            )
            data_array = concatenate_data_dictionaries(data_array, new_data_array)
            population.append(evaluated_random_children[0])

        # Remove a configuration
        if regularize:  # removing oldest, which will be first as we append new last
            killed = population.pop(0)
        else:  # removing the worst in the subset
            killed = population.pop(worst[0])
    t1 = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        ("Evolution time %10.4f sec\n" % ((t1 - t0).total_seconds()))
    )

    if profiling is not None:
        profiling.add("Evolution time", (t1 - t0).total_seconds())

    return data_array


def main(config, black_box_function=None, output_file="", profiling=None):
    """
    Run design-space exploration using evolution.
    :param config: dictionary containing all the configuration parameters of this design-space exploration.
    :param black_box_function: The function hypermapper seeks to optimize
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

    output_data_file = get_output_data_file(
        config["output_data_file"], run_directory, application_name
    )
    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)
    if number_of_objectives != 1:
        print(
            "the evolutionary optimization does not support multi-objective optimization. Exiting."
        )
        sys.exit()

    fitness_measure = optimization_metrics[0]
    population_size = config["evolution_population_size"]
    generations = config["evolution_generations"]
    mutation_rate = config["mutation_rate"]
    if mutation_rate > len(param_space.get_input_parameters()):
        print("mutation rate cannot be higher than the number of parameters. Exiting.")
        sys.exit()
    if mutation_rate < 1:
        print("mutation rate must be at least 1 for evolution to work. Exiting.")
        sys.exit()
    crossover = config["evolution_crossover"]
    regularize = config["regularize_evolution"]
    batch_size = config["batch_size"]
    if batch_size > population_size:
        print("population_size must be bigger than batch_size. Exiting.")
        sys.exit()
    elif batch_size < 2 and not crossover:
        print("batch_size cannot be smaller than 2. Exiting.")
        sys.exit()
    elif batch_size < 3 and crossover:
        print("batch_size must be at least 3 when using crossover. Exiting.")
        sys.exit()

    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    sys.stdout.set_verbose_mode(config["verbose_logging"])
    if hypermapper_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    absolute_configuration_index = 0
    fast_addressing_of_data_array = {}
    evolution_fast_addressing_of_data_array = {}
    evolution_data_array = defaultdict(list)

    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = dict()
    optimization_function_parameters["hypermapper_mode"] = hypermapper_mode
    optimization_function_parameters["param_space"] = param_space
    optimization_function_parameters["beginning_of_time"] = beginning_of_time
    optimization_function_parameters["run_directory"] = run_directory
    optimization_function_parameters["output_data_file"] = output_data_file
    optimization_function_parameters["black_box_function"] = black_box_function
    optimization_function_parameters["evolution_data_array"] = evolution_data_array
    optimization_function_parameters[
        "fast_addressing_of_data_array"
    ] = evolution_fast_addressing_of_data_array

    create_output_data_file(
        output_data_file, param_space.get_input_output_and_timestamp_parameters()
    )

    print("Starting evolution...")
    evolution_t0 = datetime.datetime.now()
    all_samples = evolution(
        population_size,
        generations,
        mutation_rate,
        crossover,
        regularize,
        batch_size,
        fitness_measure,
        param_space,
        fast_addressing_of_data_array,
        run_objective_function,
        optimization_function_parameters,
        profiling,
    )

    print(
        "Evolution finished after %d function evaluations"
        % (len(evolution_data_array[optimization_metrics[0]]))
    )
    sys.stdout.write_to_logfile(
        (
            "Evolutionary search time %10.4f sec\n"
            % ((datetime.datetime.now() - evolution_t0).total_seconds())
        )
    )

    print("### End of the evolutionary search")
    return evolution_data_array
