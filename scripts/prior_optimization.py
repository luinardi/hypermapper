###############################################################################################################################
# This script implements our Prior-guided Bayesian Optimization method, presented in: https://arxiv.org/abs/1805.12168.       #
###############################################################################################################################
import sys
import os
import space
import models
import numpy as np
import csv
import random
import json
import datetime
from jsonschema import Draft4Validator, validators, exceptions
from utility_functions import *
from local_search import local_search
from random_scalarizations import sample_weight_flat, compute_data_array_scalarization
from sklearn.ensemble import ExtraTreesRegressor

def compute_probability_from_prior(configurations, param_space, objective_weights):
    """
    Compute the probability of configurations being good according to the prior.
    :param configurations: list of configurations to compute probability.
    :param param_space: Space object for the optimization problem
    :param objective_weights: Objective weights for multi-objective optimization. Not implemented yet.
    :return: list with the probability of each configuration being good according to the prior.
    """
    probabilities = []
    objectives = param_space.get_optimization_parameters()
    input_param_objects = param_space.get_input_parameters_objects()

    prior_estimation_flag = param_space.get_estimate_prior_flags()[0]

    # We have to update this for multiple objectives
    if prior_estimation_flag:
        for configuration in configurations:
            probability = param_space.get_configuration_probability(configuration)
            probabilities.append(probability)
    else:
        for configuration in configurations:
            probability = 1
            for parameter_name in configuration.keys():
                for objective in objectives:
                    parameter_value = configuration[parameter_name]
                    p = input_param_objects[parameter_name].get_x_probability(parameter_value)
                    probability *= p**objective_weights[objective]
            probabilities.append(probability)

    return probabilities

def estimate_prior_limits(param_space, prior_limit_estimation_points, objective_weights):
    """
    Estimate the limits for the priors provided. Limits are used to normalize the priors, if prior normalization is required.
    :param param_space: Space object for the optimization problem
    :param prior_limit_estimation_points: number of points to sample to estimate the limits
    :param objective_weights: Objective weights for multi-objective optimization. Not implemented yet.
    :return: list with the estimated lower and upper limits found for the prior.
    """
    uniform_configurations = param_space.random_sample_configurations_without_repetitions({}, prior_limit_estimation_points, use_priors=False)
    prior_configurations = param_space.random_sample_configurations_without_repetitions({}, prior_limit_estimation_points, use_priors=True) # will be uniform random if no prior
    configurations = uniform_configurations + prior_configurations

    prior = compute_probability_from_prior(configurations, param_space, objective_weights)

    return [min(prior), max(prior)]

def compute_EI_from_posteriors(
                            configurations,
                            param_space,
                            objective_weights,
                            objective_limits,
                            threshold,
                            iteration_number,
                            model_weight,
                            regression_models,
                            classification_model,
                            model_type,
                            good_prior_normalization_limits,
                            tree_means_per_leaf=None,
                            tree_vars_per_leaf=None,
                            posterior_floor=10**-8,
                            posterior_normalization_limits=None,
                            debug=False):
    """
    Compute EI acquisition function for a list of configurations based on the priors provided by the user and the BO model.
    :param configurations: list of configurations to compute EI.
    :param param_space: Space object for the optimization problem
    :param objective_weights: objective weights for multi-objective optimization. Not implemented yet.
    :param objective_limits: objective limits for multi-objective optimization. Not implemented yet.
    :param threshold: threshold that separates configurations into good or bad for the model.
    :param iteration_number: current optimization iteration.
    :param model_weight: weight hyperparameter given to the model during posterior computation.
    :param regression_models: regression models to compute the probability of a configuration being good according to BO's model.
    :param classification_model: classification model to compute the probability of feasibility.
    :param model_type: type of the regression model, either GP or RF for now.
    :param good_prior_normalization_limits: lower and upper limits to normalize the prior. Will be updated if any value exceeds the limits.
    :param tree_means_per_leaf: list of number_of_trees dictionaries. Each dictionary contains the mean for each leaf in a tree.
    :param tree_vars_per_leaf: list of number_of_trees dictionaries. Each dictionary contains the variance for each leaf in a tree.
    :param posterior_floor: lower limit for posterior computation. Used when normalizing the priors and in the probability of feasibility.
    :param posterior_normalization_limits: 
    :param debug: whether to run in debug mode.
    """
    user_prior_t0 = datetime.datetime.now()
    prior_good = compute_probability_from_prior(configurations, param_space, objective_weights)

    # if prior is non-normalized, we have to normalize it
    if good_prior_normalization_limits is not None:
        good_prior_normalization_limits[0] = min(good_prior_normalization_limits[0], min(prior_good))
        good_prior_normalization_limits[1] = max(good_prior_normalization_limits[1], max(prior_good))

        # limits will be equal if all values are the same, in this case, just set the prior to 1 everywhere
        if good_prior_normalization_limits[0] == good_prior_normalization_limits[1]: 
            prior_good = [1]*len(prior_good) 
        else:
            prior_good = [posterior_floor + ((1-posterior_floor)*(x - good_prior_normalization_limits[0]))/(good_prior_normalization_limits[1] - good_prior_normalization_limits[0]) \
                        for x in prior_good]

    prior_good = np.array(prior_good, dtype=np.float64)
    prior_bad = np.array(1 - prior_good, dtype=np.float64)
    prior_bad[prior_bad < posterior_floor] = posterior_floor

    sys.stdout.write_to_logfile(("EI: user prior time %10.4f sec\n" % ((datetime.datetime.now() - user_prior_t0).total_seconds())))

    model_t0 = datetime.datetime.now()
    bufferx = dict_list_to_matrix(configurations) # prediction methods require a matrix instead of list of dictionaries
    number_of_predictions = len(bufferx)
    model_means = {}
    model_stds = {}

    if model_type == "random_forest":
        leaf_per_sample = {}
        model_variances = {}
        if (tree_means_per_leaf is None) or (tree_vars_per_leaf is None):
            print("Error: means and vars per leaf are required to compute posterior with random forests")
            raise SystemExit

        # Compute mean and variances for the model
        leaf_per_sample = models.get_leaves_per_sample(bufferx, regression_models, param_space)
        for objective in regression_models:
            model_means[objective] = models.compute_rf_prediction(leaf_per_sample[objective], tree_means_per_leaf[objective])
            model_variances[objective] = models.compute_rf_prediction_variance(
                                                                        leaf_per_sample[objective],
                                                                        model_means[objective],
                                                                        tree_means_per_leaf[objective],
                                                                        tree_vars_per_leaf[objective])
            model_stds[objective] = np.sqrt(model_variances[objective])
    elif model_type == "gaussian_process":
        model_means, model_stds = models.compute_gp_prediction_mean_and_std(bufferx, regression_models, param_space)
    else:
        print("Model not supported:", model_type)
        raise SystemExit

    # If classification model is trained, there are feasibility constraints
    if classification_model != None:
        classification_prediction_results = models.model_probabilities(bufferx, classification_model, param_space)
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = classification_model[feasible_parameter].classes_.tolist().index(True) # predictor gives both probabilities (feasible and infeasible), find the index of feasible probabilities
        feasibility_indicator = classification_prediction_results[feasible_parameter][:,true_value_index] 
        feasibility_indicator[feasibility_indicator == 0] = posterior_floor
        feasibility_indicator = np.log(feasibility_indicator)

        # Normalize the feasibility indicator to 0, 1. 
        feasibility_indicator = [posterior_floor + ((1-posterior_floor)*(x - np.log(posterior_floor)))/(np.log(1) - np.log(posterior_floor)) \
                                  for x in feasibility_indicator]
        feasibility_indicator = np.array(feasibility_indicator)

    else:
        feasibility_indicator = [1]*number_of_predictions # if classification model is not trained, all points are feasible
    
    model_good = models.compute_probability_from_model(
                                            model_means,
                                            model_stds,
                                            param_space,
                                            objective_weights,
                                            threshold,
                                            compute_bad=False)
    model_good = np.array(model_good, dtype=np.float64)

    model_bad = models.compute_probability_from_model(
                                            model_means,
                                            model_stds,
                                            param_space,
                                            objective_weights,
                                            threshold,
                                            compute_bad=True)
    sys.stdout.write_to_logfile(("EI: model time %10.4f sec\n" % ((datetime.datetime.now() - model_t0).total_seconds())))
    posterior_t0 = datetime.datetime.now()
    good_bad_ratios = np.zeros(len(configurations), dtype=np.float64)

    with np.errstate(divide='ignore'):
        log_posterior_good = np.log(prior_good) + (iteration_number/model_weight)*np.log(model_good)
        log_posterior_bad = np.log(prior_bad) + (iteration_number/model_weight)*np.log(model_bad)

    good_bad_ratios = log_posterior_good - log_posterior_bad

    # If we have feasibility constraints, normalize good_bad_ratios to 0, 1
    if posterior_normalization_limits is not None:
        tmp_gbr = copy.deepcopy(good_bad_ratios)
        tmp_gbr = np.array(tmp_gbr)
        
        # Do not consider -inf and +inf when computing the limits
        tmp_gbr[tmp_gbr == float("-inf")] = float("inf")
        posterior_normalization_limits[0] = min(posterior_normalization_limits[0], min(tmp_gbr))
        tmp_gbr[tmp_gbr == float("inf")] = float("-inf")
        posterior_normalization_limits[1] = max(posterior_normalization_limits[1], max(tmp_gbr))
        
        # limits will be equal if all values are the same, in this case, just set the prior to 1 everywhere
        if posterior_normalization_limits[0] == posterior_normalization_limits[1]:  
            good_bad_ratios = [1]*len(good_bad_ratios) 
        else:
            new_gbr = []
            for x in good_bad_ratios:
                new_x = posterior_floor + ((1-posterior_floor)*(x - posterior_normalization_limits[0]))/(posterior_normalization_limits[1] - posterior_normalization_limits[0])
                new_gbr.append(new_x)
            good_bad_ratios = new_gbr
        good_bad_ratios = np.array(good_bad_ratios)

    good_bad_ratios = good_bad_ratios + feasibility_indicator
    good_bad_ratios = -1*good_bad_ratios
    good_bad_ratios = list(good_bad_ratios)
    
    sys.stdout.write_to_logfile(("EI: posterior time %10.4f sec\n" % ((datetime.datetime.now() - posterior_t0).total_seconds())))
    sys.stdout.write_to_logfile(("EI: total time %10.4f sec\n" % ((datetime.datetime.now() - user_prior_t0).total_seconds())))

    # local search expects the optimized function to return the values and a feasibility indicator
    return good_bad_ratios, feasibility_indicator

def main(config, black_box_function=None, output_file=""):
    """
    Run design-space exploration using prior injection.
    :param config: dictionary containing all the configuration parameters of this design-space exploration.
    :param black_box_function: black-box function to optimize if running on default mode.
    :param output_file: a name for the file used to save the optimization results.
    """
    debug = False
    sys.stdout.write_to_logfile(str(config) + "\n")

    param_space = space.Space(config)

    random_time = datetime.datetime.now()

    run_directory = config["run_directory"]
    application_name = config["application_name"]
    hypermapper_mode = config["hypermapper_mode"]["mode"]

    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if (hypermapper_mode == 'client-server'):
        sys.stdout.switch_log_only_on_file(True)

    if hypermapper_mode == "default":
        if black_box_function == None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    input_params = param_space.get_input_parameters()
    optimization_metrics = config["optimization_objectives"]
    if len(optimization_metrics) > 1:
        print("Error: prior optimization does not support multi-objective optimization yet")
        exit()
    number_of_objectives = len(optimization_metrics)
    optimization_iterations = config["optimization_iterations"]
    evaluations_per_optimization_iteration = config["evaluations_per_optimization_iteration"]
    number_of_cpus = config["number_of_cpus"]
    if number_of_cpus > 1:
        print("Warning: this mode supports only sequential execution for now. Running on a single cpu.")
        number_of_cpus = 1
    print_importances = config["print_parameter_importance"]
    epsilon_greedy_threshold = config["epsilon_greedy_threshold"]

    if "feasible_output" in config:
        feasible_output = config["feasible_output"]
        feasible_output_name = feasible_output["name"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
        enable_feasible_predictor_grid_search_on_recall_and_precision = feasible_output["enable_feasible_predictor_grid_search_on_recall_and_precision"]
        feasible_predictor_grid_search_validation_file = feasible_output["feasible_predictor_grid_search_validation_file"]
        feasible_parameter = param_space.get_feasible_parameter()

    acquisition_function_optimizer = config["acquisition_function_optimizer"]
    if acquisition_function_optimizer == "local_search":
        local_search_random_points = config["local_search_random_points"]
        local_search_starting_points = config["local_search_starting_points"]
    elif acquisition_function_optimizer == "posterior_sampling":
        posterior_sampling_tuning_points = config["posterior_sampling_tuning_points"]
        posterior_sampling_final_samples = config["posterior_sampling_final_samples"]
        posterior_sampling_mcmc_chains = config["posterior_sampling_mcmc_chains"]
    else:
        print("Unrecognized acquisition function optimization method in the configuration file:", acquisition_function_optimizer)
        raise SystemExit

    exhaustive_search_data_array = None
    exhaustive_search_fast_addressing_of_data_array = None
    scalarization_key = config["scalarization_key"]
    scalarization_method = config["scalarization_method"]

    model_weight = config["model_posterior_weight"]
    model_good_quantile = config["model_good_quantile"]
    weight_sampling = config["weight_sampling"]
    objective_limits = {}
    for objective in optimization_metrics:
        objective_limits[objective] = [float("inf"), float("-inf")]

    number_of_doe_samples = config["design_of_experiment"]["number_of_samples"]

    model_type = config["models"]["model"]
    regression_model_parameters = {}
    if model_type == "random_forest":
        regression_model_parameters["n_estimators"] = config["models"]["number_of_trees"]
        regression_model_parameters["max_features"] = config["models"]["max_features"]
        regression_model_parameters["bootstrap"] = config["models"]["bootstrap"]
        regression_model_parameters["min_samples_split"] = config["models"]["min_samples_split"]
        tree_means_per_leaf=None
        tree_vars_per_leaf=None

    if output_file == "":
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
    else:
        output_data_file = output_file

    beginning_of_time = param_space.current_milli_time()
    absolute_configuration_index = 0

    if param_space.get_prior_normalization_flag() is True:
        prior_limit_estimation_points = config["prior_limit_estimation_points"]
        objective_weights = sample_weight_flat(optimization_metrics, 1)[0] # this will do fine for 1 objective cases, but for multi-objective optimization it might break
        good_prior_normalization_limits = estimate_prior_limits(param_space, prior_limit_estimation_points, objective_weights)
    else:
        good_prior_normalization_limits = None

    # Design of experiments/resume optimization phase
    doe_t0 = datetime.datetime.now()
    if config["resume_optimization"] == True:
        resume_data_file = config["resume_optimization_data"]
        if not resume_data_file.endswith('.csv'):
            print("Error: resume data file must be a CSV")
            raise SystemExit
        if resume_data_file == "output_samples.csv":
            resume_data_file = application_name + "_" + resume_data_file
        data_array, fast_addressing_of_data_array = param_space.load_data_file(resume_data_file, debug=False, number_of_cpus=number_of_cpus)
        absolute_configuration_index = len(data_array[list(data_array.keys())[0]]) # get the number of points evaluated in the previous run
        beginning_of_time = beginning_of_time - data_array[param_space.get_timestamp_parameter()[0]][-1] # Set the timestamp back to match the previous run
        print("Resumed optimization, number of samples = %d ......." % absolute_configuration_index)

        if absolute_configuration_index < number_of_doe_samples:
            configurations = param_space.get_doe_sample_configurations(
                                                                    fast_addressing_of_data_array,
                                                                    number_of_doe_samples-absolute_configuration_index,
                                                                    "random sampling")

            print("Design of experiment phase, number of new doe samples = %d ......." % (number_of_doe_samples - absolute_configuration_index))
            new_data_array = param_space.run_configurations(
                                                            hypermapper_mode,
                                                            configurations,
                                                            beginning_of_time,
                                                            black_box_function,
                                                            exhaustive_search_data_array,
                                                            exhaustive_search_fast_addressing_of_data_array,
                                                            run_directory)
            data_array = concatenate_data_dictionaries(
                                                    data_array,
                                                    new_data_array,
                                                    param_space.input_output_and_timestamp_parameter_names)
            absolute_configuration_index = number_of_doe_samples
            iteration_number = 1
        else:
            iteration_number = absolute_configuration_index - number_of_doe_samples + 1
    else:
        fast_addressing_of_data_array = {}
        default_configuration = param_space.get_default_or_random_configuration()
        str_data = param_space.get_unique_hash_string_from_values(default_configuration)
        fast_addressing_of_data_array[str_data] = absolute_configuration_index

        if number_of_doe_samples-1 > 0:
            configurations = param_space.get_doe_sample_configurations(fast_addressing_of_data_array, number_of_doe_samples-1, "random sampling") + [default_configuration]
        else:
            configurations = [default_configuration]

        print("Design of experiment phase, number of doe samples = %d ......." % number_of_doe_samples)
        data_array = param_space.run_configurations(
                                                    hypermapper_mode,
                                                    configurations,
                                                    beginning_of_time,
                                                    black_box_function,
                                                    exhaustive_search_data_array,
                                                    exhaustive_search_fast_addressing_of_data_array,
                                                    run_directory)
        absolute_configuration_index += number_of_doe_samples
        iteration_number = 1

    for objective in optimization_metrics:
        lower_bound = min(objective_limits[objective][0], min(data_array[objective]))
        upper_bound = max(objective_limits[objective][1], max(data_array[objective]))
        objective_limits[objective] = [lower_bound, upper_bound]

    if enable_feasible_predictor:
        # HyperMapper needs at least one valid and one invalid sample for its feasibility classifier
        # i.e. they cannot all be equal
        while are_all_elements_equal(data_array[feasible_parameter[0]]) and optimization_iterations > 0:
            print("Warning: all points are either valid or invalid, random sampling more configurations.")
            print("Number of doe samples so far:", absolute_configuration_index)
            configurations = param_space.get_doe_sample_configurations(fast_addressing_of_data_array, 1, "random sampling")
            new_data_array = param_space.run_configurations(
                                                            hypermapper_mode,
                                                            configurations,
                                                            beginning_of_time,
                                                            black_box_function,
                                                            exhaustive_search_data_array,
                                                            exhaustive_search_fast_addressing_of_data_array,
                                                            run_directory)
            data_array = concatenate_data_dictionaries(
                                                        new_data_array,
                                                        data_array,
                                                        param_space.input_output_and_timestamp_parameter_names)
            absolute_configuration_index += 1
            optimization_iterations -= 1

    print("\nEnd of doe/resume phase, the number of configuration runs is: %d\n" %absolute_configuration_index)
    sys.stdout.write_to_logfile(("DoE time %10.4f sec\n" % ((datetime.datetime.now() - doe_t0).total_seconds())))

    with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'w') as f:
        w = csv.writer(f)
        w.writerow(param_space.get_input_output_and_timestamp_parameters())
        tmp_list = [param_space.convert_types_to_string(j, data_array) for j in param_space.get_input_output_and_timestamp_parameters()]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])

    if evaluations_per_optimization_iteration > 1:
        print("Warning, number of evaluations per iteration > 1")
        print("HyperMapper's prior optimization currently does not support multiple runs per iteration, setting evaluations per iteration to 1")

    function_parameters = {}
    function_parameters["param_space"] = param_space
    function_parameters["model_weight"] = model_weight
    function_parameters["threshold"] = {}
    function_parameters["good_prior_normalization_limits"] = good_prior_normalization_limits
    function_parameters["posterior_floor"] = config["posterior_computation_lower_limit"]
    function_parameters['model_type'] = model_type
    bo_t0 = datetime.datetime.now()
    while iteration_number <= optimization_iterations:
        print("Starting optimization iteration", iteration_number)
        model_t0 = datetime.datetime.now()
        iteration_t0 = datetime.datetime.now()

        regression_models,_,_ = models.generate_mono_output_regression_models(
                                                                            data_array,
                                                                            param_space,
                                                                            input_params,
                                                                            optimization_metrics,
                                                                            1.00,
                                                                            model_type=model_type,
                                                                            number_of_cpus=number_of_cpus,
                                                                            print_importances=print_importances,
                                                                            **regression_model_parameters)

        normalized_objectives = {}
        for objective in optimization_metrics:
            if objective_limits[objective][1] == objective_limits[objective][0]:
                normalized_objectives[objective] = [0]*len(data_array[objective])
            else:
                normalized_objectives[objective] = [(x - objective_limits[objective][0]) \
                                        /(objective_limits[objective][1] - objective_limits[objective][0]) for x in data_array[objective]]
            function_parameters["threshold"][objective] = np.quantile(data_array[objective], model_good_quantile)

        if model_type == "random_forest":
            # Change splits of each node from (lower_bound + upper_bound)/2 to a uniformly sampled split in (lower_bound, upper_bound)
            bufferx = [data_array[input_param] for input_param in input_params]
            bufferx = list(map(list, list(zip(*bufferx))))
            tree_means_per_leaf = {}
            tree_vars_per_leaf = {}
            leaf_per_sample = models.get_leaves_per_sample(bufferx, regression_models, param_space)
            for objective in optimization_metrics:
                tree_means_per_leaf[objective] = models.get_mean_per_leaf(data_array[objective], leaf_per_sample[objective])
                tree_vars_per_leaf[objective] = models.get_var_per_leaf(data_array[objective], leaf_per_sample[objective])
            regression_models = models.transform_rf_using_uniform_splits(regression_models, data_array, param_space)
            function_parameters["tree_means_per_leaf"] = tree_means_per_leaf
            function_parameters["tree_vars_per_leaf"] = tree_vars_per_leaf

        classification_model = None
        if enable_feasible_predictor:
            classification_model,_,_ = models.generate_classification_model(application_name,
                                                                            param_space,
                                                                            data_array,
                                                                            input_params,
                                                                            feasible_parameter,
                                                                            1.00,
                                                                            debug,
                                                                            n_estimators=10,
                                                                            max_features=0.75,
                                                                            number_of_cpus=number_of_cpus,
                                                                            data_array_exhaustive=exhaustive_search_data_array,
                                                                            enable_feasible_predictor_grid_search_on_recall_and_precision=enable_feasible_predictor_grid_search_on_recall_and_precision,
                                                                            feasible_predictor_grid_search_validation_file=feasible_predictor_grid_search_validation_file,
                                                                            print_importances=print_importances)

        sys.stdout.write_to_logfile(("Model fitting time %10.4f sec\n" % ((datetime.datetime.now() - model_t0).total_seconds())))

        objective_weights = sample_weight_flat(optimization_metrics, 1)[0]

        data_array_scalarization, objective_limits = compute_data_array_scalarization(
                                                                                    data_array,
                                                                                    objective_weights,
                                                                                    objective_limits,
                                                                                    None,
                                                                                    scalarization_method)
        data_array[scalarization_key] = data_array_scalarization.tolist()
        epsilon = random.uniform(0,1)
        if epsilon > epsilon_greedy_threshold:
            function_parameters["objective_weights"] = objective_weights
            function_parameters["objective_limits"] = objective_limits
            function_parameters["iteration_number"] = iteration_number
            function_parameters["objective_weights"] = objective_weights
            function_parameters["regression_models"] = regression_models
            function_parameters['classification_model'] = classification_model
            if enable_feasible_predictor:
                function_parameters["posterior_normalization_limits"] = [float("inf"), float("-inf")]
            if acquisition_function_optimizer == "local_search":
                _, best_configuration = local_search(
                                                    local_search_starting_points,
                                                    local_search_random_points,
                                                    param_space,
                                                    fast_addressing_of_data_array,
                                                    False, # set feasibility to false, we handle it inside the acquisition function
                                                    compute_EI_from_posteriors,
                                                    function_parameters,
                                                    scalarization_key,
                                                    previous_points=data_array)
            else:
                print("Unrecognized acquisition function optimization method in the configuration file:", acquisition_function_optimizer)
                raise SystemExit

            str_data = param_space.get_unique_hash_string_from_values(best_configuration)
            fast_addressing_of_data_array[str_data] = absolute_configuration_index
            absolute_configuration_index += 1
        else:
            sys.stdout.write_to_logfile(str(epsilon) + " < " + str(epsilon_greedy_threshold) + " random sampling a configuration to run\n")
            best_configuration = param_space.random_sample_configurations_without_repetitions(fast_addressing_of_data_array, 1, use_priors=False)[0]

        black_box_t0 = datetime.datetime.now()
        best_configuration = [best_configuration]
        new_data_array = param_space.run_configurations(
                                                        hypermapper_mode,
                                                        best_configuration,
                                                        beginning_of_time,
                                                        black_box_function,
                                                        exhaustive_search_data_array,
                                                        exhaustive_search_fast_addressing_of_data_array,
                                                        run_directory)
        sys.stdout.write_to_logfile(("Black box time %10.4f sec\n" % ((datetime.datetime.now() - black_box_t0).total_seconds())))

        with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'a') as f:
            w = csv.writer(f)
            tmp_list = [param_space.convert_types_to_string(j, new_data_array) for j in list(param_space.get_input_output_and_timestamp_parameters())]
            tmp_list = list(zip(*tmp_list))
            for i in range(len(new_data_array[optimization_metrics[0]])):
                w.writerow(tmp_list[i])
        data_array = concatenate_data_dictionaries(
                                                new_data_array,
                                                data_array,
                                                param_space.input_output_and_timestamp_parameter_names)
        for objective in optimization_metrics:
            lower_bound = min(objective_limits[objective][0], min(data_array[objective]))
            upper_bound = max(objective_limits[objective][1], max(data_array[objective]))
            objective_limits[objective] = [lower_bound, upper_bound]

        iteration_number += 1
        sys.stdout.write_to_logfile(("BO iteration time %10.4f sec\n" % ((datetime.datetime.now() - iteration_t0).total_seconds())))

    sys.stdout.write_to_logfile(("BO total time %10.4f sec\n" % ((datetime.datetime.now() - bo_t0).total_seconds())))
    print("End of Prior Optimization")


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
        os.chdir(initial_directory)
    except:
        pass

    print("### End of the prior_optimization script.")
