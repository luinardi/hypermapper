#########################################################################################
# This script implements the Bayesian Optimization with a Prior for the Optimum method, #
# presented in: https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_701.pdf.       #
#########################################################################################
import copy
import datetime

import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
from scipy import stats

# ensure backward compatibility
try:
    from hypermapper import models
    from hypermapper.local_search import local_search
    from hypermapper.utility_functions import (
        dict_list_to_matrix,
        deal_with_relative_and_absolute_path,
    )
    from hypermapper.cma_es import cma_es
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

    from hypermapper import models
    from hypermapper.local_search import local_search
    from hypermapper.utility_functions import (
        dict_list_to_matrix,
        deal_with_relative_and_absolute_path,
    )
    from hypermapper.cma_es import cma_es


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
                    p = input_param_objects[parameter_name].get_x_probability(
                        parameter_value
                    )
                    probability *= p ** objective_weights[objective]
            probabilities.append(probability)

    return probabilities


def estimate_prior_limits(
    param_space, prior_limit_estimation_points, objective_weights
):
    """
    Estimate the limits for the priors provided. Limits are used to normalize the priors, if prior normalization is required.
    :param param_space: Space object for the optimization problem
    :param prior_limit_estimation_points: number of points to sample to estimate the limits
    :param objective_weights: Objective weights for multi-objective optimization. Not implemented yet.
    :return: list with the estimated lower and upper limits found for the prior.
    """
    uniform_configurations = (
        param_space.random_sample_configurations_without_repetitions(
            {}, prior_limit_estimation_points, use_priors=False
        )
    )
    prior_configurations = param_space.random_sample_configurations_without_repetitions(
        {}, prior_limit_estimation_points, use_priors=True
    )  # will be uniform random if no prior
    configurations = uniform_configurations + prior_configurations

    prior = compute_probability_from_prior(
        configurations, param_space, objective_weights
    )

    return [min(prior), max(prior)]


def compute_probability_from_model(
    model_means, model_stds, param_space, objective_weights, threshold, compute_bad=True
):
    """
    Compute the probability of a configuration being good or bad according to the model.
    :param model_means: predicted means of the model for each configuration.
    :param model_means: predicted std of the model for each configuration.
    :param param_space: Space object for the optimization problem.
    :param objective_weights: objective weights for multi-objective optimization. Not implemented yet.
    :param threshold: threshold on objective values separating good points and bad points.
    :param compute_bad: whether to compute the probability of being good or bad.
    """
    optimization_parameters = param_space.get_optimization_parameters()
    probabilities = np.ones(len(model_means[optimization_parameters[0]]))

    for parameter in optimization_parameters:
        parameter_means = model_means[parameter]
        parameter_stds = model_stds[parameter]
        if compute_bad:
            p = 1 - stats.norm.cdf(
                (threshold[parameter] - parameter_means) / parameter_stds
            )
        else:
            p = stats.norm.cdf(
                (threshold[parameter] - parameter_means) / parameter_stds
            )
        probabilities *= p ** objective_weights[parameter]

    return probabilities


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
    posterior_floor=10**-8,
    posterior_normalization_limits=None,
    debug=False,
):
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
    :param posterior_floor: lower limit for posterior computation. Used when normalizing the priors and in the probability of feasibility.
    :param posterior_normalization_limits:
    :param debug: whether to run in debug mode.
    """
    param_objects = param_space.get_input_parameters_objects()
    for parameter in param_space.get_input_parameters():
        param_min, param_max = (
            param_objects[parameter].get_min(),
            param_objects[parameter].get_max(),
        )
        for configuration in configurations:
            configuration[parameter] = min(configuration[parameter], param_max)
            configuration[parameter] = max(configuration[parameter], param_min)
    user_prior_t0 = datetime.datetime.now()
    prior_good = compute_probability_from_prior(
        configurations, param_space, objective_weights
    )

    # if prior is non-normalized, we have to normalize it
    if good_prior_normalization_limits is not None:
        good_prior_normalization_limits[0] = min(
            good_prior_normalization_limits[0], min(prior_good)
        )
        good_prior_normalization_limits[1] = max(
            good_prior_normalization_limits[1], max(prior_good)
        )

        # limits will be equal if all values are the same, in this case, just set the prior to 1 everywhere
        if good_prior_normalization_limits[0] == good_prior_normalization_limits[1]:
            prior_good = [1] * len(prior_good)
        else:
            prior_good = [
                posterior_floor
                + ((1 - posterior_floor) * (x - good_prior_normalization_limits[0]))
                / (
                    good_prior_normalization_limits[1]
                    - good_prior_normalization_limits[0]
                )
                for x in prior_good
            ]

    prior_good = np.array(prior_good, dtype=np.float64)
    prior_bad = np.array(1 - prior_good, dtype=np.float64)

    prior_bad[prior_bad < posterior_floor] = posterior_floor

    discrete_space = True
    for parameter in param_space.get_input_parameters():
        if param_space.get_type(parameter) == "real":
            discrete_space = False

    if discrete_space:
        prior_bad = prior_bad / (param_space.get_discrete_space_size() - 1)

    sys.stdout.write_to_logfile(
        (
            "EI: user prior time %10.4f sec\n"
            % ((datetime.datetime.now() - user_prior_t0).total_seconds())
        )
    )

    model_t0 = datetime.datetime.now()
    bufferx = dict_list_to_matrix(
        configurations
    )  # prediction methods require a matrix instead of list of dictionaries
    number_of_predictions = len(bufferx)
    model_stds = {}

    model_means, model_stds = models.compute_model_mean_and_uncertainty(
        bufferx, regression_models, model_type, param_space, var=False
    )

    # If classification model is trained, there are feasibility constraints
    if classification_model != None:
        classification_prediction_results = models.model_probabilities(
            bufferx, classification_model, param_space
        )
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = (
            classification_model[feasible_parameter].classes_.tolist().index(True)
        )  # predictor gives both probabilities (feasible and infeasible), find the index of feasible probabilities
        feasibility_indicator = classification_prediction_results[feasible_parameter][
            :, true_value_index
        ]
        feasibility_indicator[feasibility_indicator == 0] = posterior_floor
        feasibility_indicator = np.log(feasibility_indicator)

        # Normalize the feasibility indicator to 0, 1.
        feasibility_indicator = [
            posterior_floor
            + ((1 - posterior_floor) * (x - np.log(posterior_floor)))
            / (np.log(1) - np.log(posterior_floor))
            for x in feasibility_indicator
        ]
        feasibility_indicator = np.array(feasibility_indicator)

    else:
        feasibility_indicator = [
            1
        ] * number_of_predictions  # if classification model is not trained, all points are feasible

    model_good = compute_probability_from_model(
        model_means,
        model_stds,
        param_space,
        objective_weights,
        threshold,
        compute_bad=False,
    )
    model_good = np.array(model_good, dtype=np.float64)

    model_bad = compute_probability_from_model(
        model_means,
        model_stds,
        param_space,
        objective_weights,
        threshold,
        compute_bad=True,
    )
    sys.stdout.write_to_logfile(
        (
            "EI: model time %10.4f sec\n"
            % ((datetime.datetime.now() - model_t0).total_seconds())
        )
    )
    posterior_t0 = datetime.datetime.now()
    good_bad_ratios = np.zeros(len(configurations), dtype=np.float64)

    with np.errstate(divide="ignore"):
        log_posterior_good = np.log(prior_good) + (
            iteration_number / model_weight
        ) * np.log(model_good)
        log_posterior_bad = np.log(prior_bad) + (
            iteration_number / model_weight
        ) * np.log(model_bad)

    good_bad_ratios = log_posterior_good - log_posterior_bad

    # If we have feasibility constraints, normalize good_bad_ratios to 0, 1
    if posterior_normalization_limits is not None:
        tmp_gbr = copy.deepcopy(good_bad_ratios)
        tmp_gbr = np.array(tmp_gbr)

        # Do not consider -inf and +inf when computing the limits
        tmp_gbr[tmp_gbr == float("-inf")] = float("inf")
        posterior_normalization_limits[0] = min(
            posterior_normalization_limits[0], min(tmp_gbr)
        )
        tmp_gbr[tmp_gbr == float("inf")] = float("-inf")
        posterior_normalization_limits[1] = max(
            posterior_normalization_limits[1], max(tmp_gbr)
        )

        # limits will be equal if all values are the same, in this case, just set the prior to 1 everywhere
        if posterior_normalization_limits[0] == posterior_normalization_limits[1]:
            good_bad_ratios = [1] * len(good_bad_ratios)
        else:
            new_gbr = []
            for x in good_bad_ratios:
                new_x = posterior_floor + (
                    (1 - posterior_floor) * (x - posterior_normalization_limits[0])
                ) / (
                    posterior_normalization_limits[1]
                    - posterior_normalization_limits[0]
                )
                new_gbr.append(new_x)
            good_bad_ratios = new_gbr
        good_bad_ratios = np.array(good_bad_ratios)

    good_bad_ratios = good_bad_ratios + feasibility_indicator
    good_bad_ratios = -1 * good_bad_ratios
    good_bad_ratios[good_bad_ratios == float("inf")] = sys.maxsize
    good_bad_ratios[good_bad_ratios == float("-inf")] = -1 * sys.maxsize
    good_bad_ratios = list(good_bad_ratios)

    sys.stdout.write_to_logfile(
        (
            "EI: posterior time %10.4f sec\n"
            % ((datetime.datetime.now() - posterior_t0).total_seconds())
        )
    )
    sys.stdout.write_to_logfile(
        (
            "EI: total time %10.4f sec\n"
            % ((datetime.datetime.now() - user_prior_t0).total_seconds())
        )
    )
    # local search expects the optimized function to return the values and a feasibility indicator
    return good_bad_ratios, feasibility_indicator


def prior_guided_optimization(
    config,
    data_array,
    param_space,
    fast_addressing_of_data_array,
    regression_models,
    iteration_number,
    objective_weights,
    objective_limits,
    classification_model=None,
    profiling=None,
    acquisition_function_optimizer="local_search",
):
    """
    Run a prior-guided bayesian optimization iteration.
    :param config: dictionary containing all the configuration parameters of this optimization.
    :param data_array: a dictionary containing previously explored points and their function values.
    :param param_space: parameter space object for the current application.
    :param fast_addressing_of_data_array: dictionary for quick-access to previously explored configurations.
    :param regression_models: the surrogate models used to evaluate points.
    :param iteration_number: the current iteration number.
    :param objective_weights: objective weights for multi-objective optimization. Not implemented yet.
    :param objective_limits: estimated minimum and maximum limits for each objective.
    :param classification_model: feasibility classifier for constrained optimization.
    """
    scalarization_key = config["scalarization_key"]
    number_of_cpus = config["number_of_cpus"]
    function_parameters = {}
    function_parameters["param_space"] = param_space
    function_parameters["iteration_number"] = iteration_number
    function_parameters["regression_models"] = regression_models
    function_parameters["classification_model"] = classification_model
    function_parameters["objective_weights"] = objective_weights
    function_parameters["objective_limits"] = objective_limits
    function_parameters["model_type"] = config["models"]["model"]
    function_parameters["model_weight"] = config["model_posterior_weight"]
    function_parameters["posterior_floor"] = config["posterior_computation_lower_limit"]
    model_good_quantile = config["model_good_quantile"]
    function_parameters["threshold"] = {}
    optimization_metrics = param_space.get_optimization_parameters()
    for objective in optimization_metrics:
        function_parameters["threshold"][objective] = np.quantile(
            data_array[objective], model_good_quantile
        )

    if param_space.get_prior_normalization_flag() is True:
        prior_limit_estimation_points = config["prior_limit_estimation_points"]
        good_prior_normalization_limits = estimate_prior_limits(
            param_space, prior_limit_estimation_points, objective_weights
        )
    else:
        good_prior_normalization_limits = None
    function_parameters[
        "good_prior_normalization_limits"
    ] = good_prior_normalization_limits

    if classification_model is not None:
        function_parameters["posterior_normalization_limits"] = [
            float("inf"),
            float("-inf"),
        ]

    if acquisition_function_optimizer == "local_search":
        local_search_starting_points = config["local_search_starting_points"]
        local_search_random_points = config["local_search_random_points"]
        _, best_configuration = local_search(
            local_search_starting_points,
            local_search_random_points,
            param_space,
            fast_addressing_of_data_array,
            False,  # set feasibility to false, we handle it inside the acquisition function
            compute_EI_from_posteriors,
            function_parameters,
            scalarization_key,
            number_of_cpus,
            previous_points=data_array,
            profiling=profiling,
        )
    elif acquisition_function_optimizer == "cma_es":
        logfile = deal_with_relative_and_absolute_path(
            config["run_directory"], config["log_file"]
        )
        sigma = config["cma_es_sigma"]
        cma_es_starting_points = config["cma_es_starting_points"]
        cma_es_random_points = config["cma_es_random_points"]
        best_configuration = cma_es(
            param_space,
            data_array,
            fast_addressing_of_data_array,
            scalarization_key,
            logfile,
            compute_EI_from_posteriors,
            function_parameters,
            cma_es_random_points=cma_es_random_points,
            cma_es_starting_points=cma_es_starting_points,
            sigma=sigma,
        )
    else:
        print(
            "Unrecognized acquisition function optimizer:",
            acquisition_function_optimizer,
        )
        raise SystemExit

    return best_configuration
