###############################################################################################################################
# This script implements an adaptation of the optimization method proposed by Paria et al.: https://arxiv.org/abs/1805.12168. #
# Our adaptations to the original are:                                                                                        #
# A different tchebyshev scalarization function                                                                               #
# A RF model instead of GP                                                                                                    #
# A multi-start local search to optimize the acquisition functions instead of the DIRECT algorithm                            #
# a contrained optimization implementation as proposed by Gardner et al. http://proceedings.mlr.press/v32/gardner14.pdf       #
###############################################################################################################################
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
        concatenate_list_of_dictionaries,
        data_dictionary_to_tuple,
        reciprocate_weights,
        compute_data_array_scalarization,
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
        concatenate_list_of_dictionaries,
        data_dictionary_to_tuple,
        reciprocate_weights,
        compute_data_array_scalarization,
        deal_with_relative_and_absolute_path,
    )
    from hypermapper.cma_es import cma_es


def run_acquisition_function(
    acquisition_function,
    configurations,
    objective_weights,
    regression_models,
    param_space,
    scalarization_method,
    objective_limits,
    iteration_number,
    data_array,
    model_type,
    classification_model=None,
    number_of_cpus=0,
):
    """
    Apply the chosen acquisition function to a list of configurations.
    :param acquisition_function: a string defining which acquisition function to apply
    :param bufferx: a list of tuples containing the configurations.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param evaluations_per_optimization_iteration: how many configurations to return.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param iteration_number: an integer for the current iteration number, used to compute the beta on ucb
    :param classification_model: the surrogate model used to evaluate feasibility constraints
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    tmp_objective_limits = None
    configurations = concatenate_list_of_dictionaries(configurations)
    configurations = data_dictionary_to_tuple(
        configurations, param_space.get_input_parameters()
    )
    if acquisition_function == "TS":
        scalarized_values, tmp_objective_limits = thompson_sampling(
            configurations,
            objective_weights,
            regression_models,
            param_space,
            scalarization_method,
            objective_limits,
            model_type,
            classification_model,
            number_of_cpus,
        )
    elif acquisition_function == "UCB":
        scalarized_values, tmp_objective_limits = ucb(
            configurations,
            objective_weights,
            regression_models,
            param_space,
            scalarization_method,
            objective_limits,
            iteration_number,
            model_type,
            classification_model,
            number_of_cpus,
        )
    elif acquisition_function == "EI":
        scalarized_values, tmp_objective_limits = EI(
            configurations,
            data_array,
            objective_weights,
            regression_models,
            param_space,
            scalarization_method,
            objective_limits,
            iteration_number,
            model_type,
            classification_model,
            number_of_cpus,
        )
    else:
        print("Unrecognized acquisition function:", acquisition_function)
        raise SystemExit

    scalarized_values = list(scalarized_values)

    # we want the local search to consider all points feasible, we already account for feasibility it in the scalarized value
    feasibility_indicators = [1] * len(scalarized_values)

    return scalarized_values, feasibility_indicators


def ucb(
    bufferx,
    objective_weights,
    regression_models,
    param_space,
    scalarization_method,
    objective_limits,
    iteration_number,
    model_type,
    classification_model=None,
    number_of_cpus=0,
):
    """
    Multi-objective ucb acquisition function as detailed in https://arxiv.org/abs/1805.12168.
    The mean and variance of the predictions are computed as defined by Hutter et al.: https://arxiv.org/pdf/1211.0906.pdf
    :param bufferx: a list of tuples containing the points to predict and scalarize.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param evaluations_per_optimization_iteration: how many configurations to return.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param iteration_number: an integer for the current iteration number, used to compute the beta
    :param classification_model: the surrogate model used to evaluate feasibility constraints
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    beta = np.sqrt(0.125 * np.log(2 * iteration_number + 1))
    augmentation_constant = 0.05
    prediction_means = {}
    prediction_variances = {}
    number_of_predictions = len(bufferx)
    tmp_objective_limits = copy.deepcopy(objective_limits)

    prediction_means, prediction_variances = models.compute_model_mean_and_uncertainty(
        bufferx, regression_models, model_type, param_space, var=True
    )
    if classification_model != None:
        classification_prediction_results = models.model_probabilities(
            bufferx, classification_model, param_space
        )
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = (
            classification_model[feasible_parameter].classes_.tolist().index(True)
        )
        feasibility_indicator = classification_prediction_results[feasible_parameter][
            :, true_value_index
        ]
    else:
        feasibility_indicator = [
            1
        ] * number_of_predictions  # if no classification model is used, then all points are feasible

    # Compute scalarization
    if scalarization_method == "linear":
        scalarized_predictions = np.zeros(number_of_predictions)
        beta_factor = 0
        for objective in regression_models:
            scalarized_predictions += (
                objective_weights[objective] * prediction_means[objective]
            )
            beta_factor += (
                objective_weights[objective] * prediction_variances[objective]
            )
        scalarized_predictions -= beta * np.sqrt(beta_factor)
        scalarized_predictions = scalarized_predictions * feasibility_indicator
    # The paper does not propose this, I applied their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif scalarization_method == "tchebyshev":
        scalarized_predictions = np.zeros(number_of_predictions)
        total_values = np.zeros(number_of_predictions)
        for objective in regression_models:
            scalarized_values = objective_weights[objective] * (
                prediction_means[objective]
                - beta * np.sqrt(prediction_variances[objective])
            )
            total_values += scalarized_values
            scalarized_predictions = np.maximum(
                scalarized_values, scalarized_predictions
            )
        scalarized_predictions += augmentation_constant * total_values
        scalarized_predictions = scalarized_predictions * feasibility_indicator
    elif scalarization_method == "modified_tchebyshev":
        scalarized_predictions = np.full((number_of_predictions), float("inf"))
        reciprocated_weights = reciprocate_weights(objective_weights)
        for objective in regression_models:
            scalarized_value = reciprocated_weights[objective] * (
                prediction_means[objective]
                - beta * np.sqrt(prediction_variances[objective])
            )
            scalarized_predictions = np.minimum(
                scalarized_value, scalarized_predictions
            )
        scalarized_predictions = scalarized_predictions * feasibility_indicator
        scalarized_predictions = (
            -scalarized_predictions
        )  # We will minimize later, but we want to maximize instead, so we invert the sign
    else:
        print("Error: unrecognized scalarization method:", scalarization_method)
        raise SystemExit

    return scalarized_predictions, tmp_objective_limits


def thompson_sampling(
    bufferx,
    objective_weights,
    regression_models,
    param_space,
    scalarization_method,
    objective_limits,
    model_type,
    classification_model=None,
    number_of_cpus=0,
):
    """
    Multi-objective thompson sampling acquisition function as detailed in https://arxiv.org/abs/1805.12168.
    :param bufferx: a list of tuples containing the points to predict and scalarize.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    tmp_objective_limits = copy.deepcopy(objective_limits)
    model_predictions = {}

    t0 = datetime.datetime.now()
    model_predictions = models.sample_model_posterior(
        bufferx, regression_models, model_type, param_space
    )
    number_of_predictions = len(model_predictions[list(model_predictions.keys())[0]])

    if classification_model != None:
        classification_prediction_results = models.model_probabilities(
            bufferx, classification_model, param_space
        )
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = (
            classification_model[feasible_parameter].classes_.tolist().index(True)
        )
        feasibility_indicator = classification_prediction_results[feasible_parameter][
            :, true_value_index
        ]
    else:
        feasibility_indicator = [
            1
        ] * number_of_predictions  # if no classification model is used, then all points are feasible

    if scalarization_method == "linear":
        scalarized_predictions = np.zeros(number_of_predictions)
        for objective in regression_models:
            scalarized_predictions += (
                objective_weights[objective] * model_predictions[objective]
            )
        scalarized_predictions = scalarized_predictions * feasibility_indicator
    # The paper does not propose this, I applied their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif scalarization_method == "tchebyshev":
        scalarized_predictions = np.zeros(number_of_predictions)
        scalarized_values = np.zeros(number_of_predictions)
        total_values = np.zeros(number_of_predictions)
        for objective in regression_models:
            scalarized_values = objective_weights[objective] * np.absolute(
                model_predictions[objective]
            )
            total_values += scalarized_values
            scalarized_predictions = np.maximum(
                scalarized_values, scalarized_predictions
            )
        scalarized_predictions += 0.05 * total_values
        scalarized_predictions = scalarized_predictions * feasibility_indicator
    elif scalarization_method == "modified_tchebyshev":
        scalarized_predictions = np.full((number_of_predictions), float("inf"))
        reciprocated_weights = reciprocate_weights(objective_weights)
        for objective in regression_models:
            scalarized_value = reciprocated_weights[objective] * np.absolute(
                model_predictions[objective]
            )
            # TODO scalarized_predictions2 not defined
            scalarized_predictions = np.minimum(
                scalarized_value, scalarized_predictions
            )
        scalarized_predictions = scalarized_predictions * feasibility_indicator
        scalarized_predictions = (
            -scalarized_predictions
        )  # We will minimize later, but we want to maximize instead, so we invert the sign
    else:
        print("Error: unrecognized scalarization method:", scalarization_method)
        raise SystemExit

    return scalarized_predictions, tmp_objective_limits


def EI(
    bufferx,
    data_array,
    objective_weights,
    regression_models,
    param_space,
    scalarization_method,
    objective_limits,
    iteration_number,
    model_type,
    classification_model=None,
    number_of_cpus=0,
):
    """
    Compute a multi-objective EI acquisition function on bufferx.
    The mean and variance of the predictions are computed as defined by Hutter et al.: https://arxiv.org/pdf/1211.0906.pdf
    :param bufferx: a list of tuples containing the points to predict and scalarize.
    :param data_array: a dictionary containing the previously run points and their function values.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param evaluations_per_optimization_iteration: how many configurations to return.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param iteration_number: an integer for the current iteration number, used to compute the beta
    :param classification_model: the surrogate model used to evaluate feasibility constraints
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    augmentation_constant = 0.05
    prediction_means = {}
    prediction_variances = {}
    number_of_predictions = len(bufferx)
    tmp_objective_limits = copy.deepcopy(objective_limits)

    prediction_means, prediction_variances = models.compute_model_mean_and_uncertainty(
        bufferx, regression_models, model_type, param_space, var=True
    )

    if classification_model != None:
        classification_prediction_results = models.model_probabilities(
            bufferx, classification_model, param_space
        )
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = (
            classification_model[feasible_parameter].classes_.tolist().index(True)
        )
        feasibility_indicator = classification_prediction_results[feasible_parameter][
            :, true_value_index
        ]
    else:
        feasibility_indicator = [
            1
        ] * number_of_predictions  # if no classification model is used, then all points are feasible

    data_array_scalarization, tmp_objective_limits = compute_data_array_scalarization(
        data_array, objective_weights, tmp_objective_limits, scalarization_method
    )

    if scalarization_method == "linear":
        scalarized_predictions = np.zeros(number_of_predictions)
        scalarized_value = 0
        for objective in regression_models:
            if (
                tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0]
                == 0
            ):
                difference_obj = 1
            else:
                difference_obj = (
                    tmp_objective_limits[objective][1]
                    - tmp_objective_limits[objective][0]
                )
            f_min = 1 - (
                min(data_array[objective]) - tmp_objective_limits[objective][0]
            ) / (difference_obj)
            x_std = np.sqrt(prediction_variances[objective])
            x_mean = 1 - prediction_means[objective]
            v = (x_mean - f_min) / x_std
            objective_ei = (x_mean - f_min) * stats.norm.cdf(
                v
            ) + x_std * stats.norm.pdf(v)
            scalarized_predictions += objective_ei * objective_weights[objective]
        scalarized_predictions = -1 * scalarized_predictions * feasibility_indicator
    # The paper does not propose this, I applied their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif scalarization_method == "tchebyshev":
        scalarized_predictions = np.zeros(number_of_predictions)
        total_value = np.zeros(number_of_predictions)
        for objective in regression_models:
            if (
                tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0]
                == 0
            ):
                difference_obj = 1
            else:
                difference_obj = (
                    tmp_objective_limits[objective][1]
                    - tmp_objective_limits[objective][0]
                )
            f_min = 1 - (
                min(data_array[objective]) - tmp_objective_limits[objective][0]
            ) / (difference_obj)
            x_std = np.sqrt(prediction_variances[objective])
            x_mean = 1 - prediction_means[objective]
            v = (x_mean - f_min) / x_std
            scalarized_value = objective_weights[objective] * (
                (1 - prediction_means[objective] - f_min) * stats.norm.cdf(v)
                + x_std * stats.norm.pdf(v)
            )
            scalarized_predictions = np.maximum(
                scalarized_value, scalarized_predictions
            )
            total_value += scalarized_value
        scalarized_predictions = (
            -1
            * (scalarized_predictions + total_value * augmentation_constant)
            * feasibility_indicator
        )
    elif scalarization_method == "modified_tchebyshev":
        scalarized_predictions = np.full((number_of_predictions), float("inf"))
        reciprocated_weights = reciprocate_weights(objective_weights)
        for objective in regression_models:
            if (
                tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0]
                == 0
            ):
                difference_obj = 1
            else:
                difference_obj = (
                    tmp_objective_limits[objective][1]
                    - tmp_objective_limits[objective][0]
                )
            f_min = 1 - (
                min(data_array[objective]) - tmp_objective_limits[objective][0]
            ) / (difference_obj)
            x_std = np.sqrt(prediction_variances[objective])
            x_mean = 1 - prediction_means[objective]
            v = (x_mean - f_min) / x_std
            scalarized_value = reciprocated_weights[objective] * (
                (x_mean - f_min) * stats.norm.cdf(v) + x_std * stats.norm.pdf(v)
            )
            scalarized_predictions = np.minimum(
                scalarized_value, scalarized_predictions
            )
        scalarized_predictions = scalarized_predictions * feasibility_indicator
    else:
        print("Error: unrecognized scalarization method:", scalarization_method)
        raise SystemExit

    return scalarized_predictions, tmp_objective_limits


def random_scalarizations(
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
    Run one iteration of bayesian optimization with random scalarizations.
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
    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)
    local_search_starting_points = config["local_search_starting_points"]
    local_search_random_points = config["local_search_random_points"]
    scalarization_key = config["scalarization_key"]
    number_of_cpus = config["number_of_cpus"]

    optimization_function_parameters = {}
    optimization_function_parameters["regression_models"] = regression_models
    optimization_function_parameters["iteration_number"] = iteration_number
    optimization_function_parameters["data_array"] = data_array
    optimization_function_parameters["classification_model"] = classification_model
    optimization_function_parameters["param_space"] = param_space
    optimization_function_parameters["objective_weights"] = objective_weights
    optimization_function_parameters["model_type"] = config["models"]["model"]
    optimization_function_parameters["objective_limits"] = objective_limits
    optimization_function_parameters["acquisition_function"] = config[
        "acquisition_function"
    ]
    optimization_function_parameters["scalarization_method"] = config[
        "scalarization_method"
    ]
    optimization_function_parameters["number_of_cpus"] = config["number_of_cpus"]

    if acquisition_function_optimizer == "local_search":
        _, best_configuration = local_search(
            local_search_starting_points,
            local_search_random_points,
            param_space,
            fast_addressing_of_data_array,
            False,  # we do not want the local search to consider feasibility constraints, only the acquisition functions
            run_acquisition_function,
            optimization_function_parameters,
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
            run_acquisition_function,
            optimization_function_parameters,
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
