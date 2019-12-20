###############################################################################################################################
# This script implements an adaptation of the optimization method proposed by Paria et al.: https://arxiv.org/abs/1805.12168. #
# Our adaptations to the original are:                                                                                        #
# A different tchebyshev scalarization function                                                                               #
# A RF model instead of GP                                                                                                    #
# A multi-start local search to optimize the acquisition functions instead of the DIRECT algorithm                            #
# a contrained optimization implementation as proposed by Gardner et al. http://proceedings.mlr.press/v32/gardner14.pdf       #
###############################################################################################################################
import sys
import os
import space
import random
import models
from sklearn.ensemble import ExtraTreesRegressor
import operator
import numpy as np
import csv
import json
import copy
import datetime
from jsonschema import Draft4Validator, validators, exceptions
from utility_functions import *
from collections import defaultdict
from scipy import stats
from local_search import get_min_configurations, get_neighbors, local_search

def sample_weight_bbox(optimization_metrics, objective_bounds, evaluations_per_optimization_iteration):
    """
    Sample lambdas for each objective following a uniform distribution with user-defined bounding boxes.
    If the user does not define bounding boxes, it defaults to [0, 1].
    :param optimization_metrics: a list containing the optimization objectives.
    :param objective_bounds: a dictionary containing the bounding boxes for each objective.
    :param evaluations_per_optimization_iteration: number of weight arrays to sample. Currently not used.
    :return: a dictionary containing the weight of each objective.
    """
    weight_list = []
    for run_idx in range(evaluations_per_optimization_iteration):
        objective_weights = {}
        for objective in optimization_metrics:
            loc, scale = objective_bounds[objective]
            scale = scale - loc # scipy.stats automatically does scale = scale + loc, we don't want that
            objective_weight = stats.uniform.rvs(loc=loc, scale=scale)
            objective_weights[objective] = objective_weight
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
    sampled_weights = stats.dirichlet.rvs(alpha=alphas, size=evaluations_per_optimization_iteration)
    weight_list = []

    for run_idx in range(evaluations_per_optimization_iteration):
        objective_weights = {}
        for idx, objective in enumerate(optimization_metrics):
            objective_weights[objective] = sampled_weights[run_idx][idx]
        weight_list.append(objective_weights)

    return weight_list

def reciprocate_weights(objective_weights):
    """
    Reciprocate weights so that they correlate when using modified_tchebyshev scalarization.
    :param objective_weights: a dictionary containing the weights for each objective.
    :return: a dictionary containing the reciprocated weights.
    """
    new_weights = {}
    total_weight = 0
    for objective in objective_weights:
        new_weights[objective] = 1/objective_weights[objective]
        total_weight += new_weights[objective]

    for objective in new_weights:
        new_weights[objective] = new_weights[objective]/total_weight

    return new_weights

def run_acquisition_function(acquisition_function,
                            configurations,
                            objective_weights,
                            regression_models,
                            param_space,
                            scalarization_method,
                            objective_bounds,
                            objective_limits,
                            iteration_number,
                            data_array,
                            classification_model=None,
                            tree_means_per_leaf=None,
                            tree_vars_per_leaf=None,
                            number_of_cpus=0):
    """
    Apply the chosen acquisition function to a list of configurations.
    :param acquisition_function: a string defining which acquisition function to apply
    :param bufferx: a list of tuples containing the configurations.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param evaluations_per_optimization_iteration: how many configurations to return.
    :param objective_bounds: a list containing bounds for the objectives bounding box.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param iteration_number: an integer for the current iteration number, used to compute the beta on ucb
    :param classification_model: the surrogate model used to evaluate feasibility constraints
    :param tree_means_per_leaf: prediction mean for each leaf of each tree.
    :param tree_vars_per_leaf: prediction variance for each leaf of each tree.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    tmp_objective_limits = None
    configurations = concatenate_list_of_dictionaries(configurations)
    configurations = data_dictionary_to_tuple(configurations, param_space.get_input_parameters())
    if acquisition_function == "TS":
        scalarized_values, tmp_objective_limits = thompson_sampling(
                                                                    configurations,
                                                                    objective_weights,
                                                                    regression_models,
                                                                    param_space,
                                                                    scalarization_method,
                                                                    objective_bounds,
                                                                    objective_limits,
                                                                    classification_model,
                                                                    number_of_cpus)
    elif acquisition_function == "UCB":
        scalarized_values, tmp_objective_limits = ucb(
                                                        configurations,
                                                        objective_weights,
                                                        regression_models,
                                                        param_space,
                                                        scalarization_method,
                                                        objective_bounds,
                                                        objective_limits,
                                                        iteration_number,
                                                        classification_model,
                                                        tree_means_per_leaf,
                                                        tree_vars_per_leaf,
                                                        number_of_cpus)
    elif acquisition_function == "EI":
        scalarized_values, tmp_objective_limits = EI(configurations,
                                                        data_array,
                                                        objective_weights,
                                                        regression_models,
                                                        param_space,
                                                        scalarization_method,
                                                        objective_bounds,
                                                        objective_limits,
                                                        iteration_number,
                                                        classification_model,
                                                        tree_means_per_leaf,
                                                        tree_vars_per_leaf,
                                                        number_of_cpus)
    else:
        print("Unrecognized acquisition function:", acquisition_function)
        raise SystemExit

    scalarized_values = list(scalarized_values)

    # we want the local search to consider all points feasible, we already account for feasibility it in the scalarized value
    feasibility_indicators = [1]*len(scalarized_values)

    return scalarized_values, feasibility_indicators


def ucb(bufferx,
        objective_weights,
        regression_models,
        param_space,
        scalarization_method,
        objective_bounds,
        objective_limits,
        iteration_number,
        classification_model=None,
        tree_means_per_leaf=None,
        tree_vars_per_leaf=None,
        number_of_cpus=0):
    """
    Multi-objective ucb acquisition function as detailed in https://arxiv.org/abs/1805.12168.
    The mean and variance of the predictions are computed as defined by Hutter et al.: https://arxiv.org/pdf/1211.0906.pdf
    :param bufferx: a list of tuples containing the points to predict and scalarize.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param evaluations_per_optimization_iteration: how many configurations to return.
    :param objective_bounds: a list containing user-defined bounds for the objectives bounding box.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param iteration_number: an integer for the current iteration number, used to compute the beta
    :param classification_model: the surrogate model used to evaluate feasibility constraints
    :param tree_means_per_leaf: prediction mean for each leaf of each tree.
    :param tree_vars_per_leaf: prediction variance for each leaf of each tree.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    beta = np.sqrt(0.125*np.log(2*iteration_number + 1))
    augmentation_constant = 0.05
    prediction_means = {}
    prediction_variances = {}
    number_of_predictions = len(bufferx)
    tmp_objective_limits = copy.deepcopy(objective_limits)

    tree_predictions = models.tree_predictions(bufferx, regression_models, param_space)
    leaf_per_sample = models.get_leaves_per_sample(bufferx, regression_models, param_space)
    for objective in regression_models:
        tmp_min = np.amin(tree_predictions[objective])
        tmp_objective_limits[objective][0] = min(tmp_min, tmp_objective_limits[objective][0])
        tmp_max = np.amax(tree_predictions[objective])
        tmp_objective_limits[objective][1] = max(tmp_max, tmp_objective_limits[objective][1])

        prediction_means[objective] = models.compute_rf_prediction(leaf_per_sample[objective], tree_means_per_leaf[objective])
        prediction_variances[objective] = models.compute_rf_prediction_variance(
                                                                            leaf_per_sample[objective],
                                                                            prediction_means[objective],
                                                                            tree_means_per_leaf[objective],
                                                                            tree_vars_per_leaf[objective])


    # Normalize weights to [0, 1] and sum(weights) = 1
    total_weight = 0
    normalized_weights = {}

    if objective_bounds is not None: 
        for objective in objective_weights:
            # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
            # We cannot optimize an objective when all values are the same, so we set its weight to 0.
            if tmp_objective_limits[objective][1] == tmp_objective_limits[objective][0]:
                normalized_weights[objective] = 0
            else:
                normalized_weights[objective] = (objective_weights[objective] - tmp_objective_limits[objective][0]) \
                                                /(tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0])

            total_weight += normalized_weights[objective]
        if total_weight == 0:
            total_weight = 1

        for objective in normalized_weights:
            normalized_weights[objective] = normalized_weights[objective]/total_weight
    else:
        normalized_weights = copy.deepcopy(objective_weights)

    if classification_model != None:
        classification_prediction_results = models.model_probabilities(bufferx, classification_model, param_space)
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = classification_model[feasible_parameter].classes_.tolist().index(True)
        feasibility_indicator = classification_prediction_results[feasible_parameter][:,true_value_index]
    else:
        feasibility_indicator = [1]*number_of_predictions # if no classification model is used, then all points are feasible

    # Compute scalarization
    if (scalarization_method == "linear"):
        scalarized_predictions = np.zeros(number_of_predictions)
        for x_index in range(number_of_predictions):
            beta_factor = 0
            for objective in regression_models:
                scalarized_predictions[x_index] += normalized_weights[objective]*prediction_means[objective][x_index]
                beta_factor += normalized_weights[objective]*prediction_variances[objective][x_index]
            scalarized_predictions[x_index] -= beta*np.sqrt(beta_factor)
            scalarized_predictions[x_index] = scalarized_predictions[x_index]*feasibility_indicator[x_index]
    # The paper does not propose this, I applied their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif(scalarization_method == "tchebyshev"):
        scalarized_predictions = scalarized_predictions = np.zeros(number_of_predictions)
        for x_index in range(number_of_predictions):
            total_value = 0
            for objective in regression_models:
                scalarized_value = normalized_weights[objective] * abs(prediction_means[objective][x_index] - beta*np.sqrt(prediction_variances[objective][x_index]))
                scalarized_predictions[x_index] = max(scalarized_value, scalarized_predictions[x_index])
                total_value += scalarized_value
            scalarized_predictions[x_index] += augmentation_constant*total_value
            scalarized_predictions[x_index] = scalarized_predictions[x_index]*feasibility_indicator[x_index]
    elif(scalarization_method == "modified_tchebyshev"):
        scalarized_predictions = np.full((number_of_predictions), float("inf"))
        reciprocated_weights = reciprocate_weights(normalized_weights)
        for x_index in range(number_of_predictions):
            for objective in regression_models:
                scalarized_value = reciprocated_weights[objective] * (prediction_means[objective][x_index] - beta*np.sqrt(prediction_variances[objective][x_index]))
                scalarized_predictions[x_index] = min(scalarized_value, scalarized_predictions[x_index])
            scalarized_predictions[x_index] = scalarized_predictions[x_index]*feasibility_indicator[x_index]
            scalarized_predictions[x_index] = -scalarized_predictions[x_index] # We will minimize later, but we want to maximize instead, so we invert the sign
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
                    objective_bounds,
                    objective_limits,
                    classification_model,
                    number_of_cpus=0):
    """
    Multi-objective thompson sampling acquisition function as detailed in https://arxiv.org/abs/1805.12168.
    :param bufferx: a list of tuples containing the points to predict and scalarize.
    :param objective_weights: a list containing the weights for each objective.
    :param regression_models: the surrogate models used to evaluate points.
    :param param_space: a space object containing the search space.
    :param scalarization_method: a string indicating which scalarization method to use.
    :param objective_bounds: a list containing bounds for the objectives bounding box.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    tmp_objective_limits = copy.deepcopy(objective_limits)

    # For now, running on parallel slows down the local search because of the number of small prediction batches
    # regression_prediction_results = models.parallel_model_prediction(regression_models, bufferx, number_of_cpus=number_of_cpus)
    regression_prediction_results = models.model_prediction(bufferx, regression_models, param_space)
    number_of_predictions = len(regression_prediction_results[list(regression_prediction_results.keys())[0]])

    if classification_model != None:
        classification_prediction_results = models.model_probabilities(bufferx, classification_model, param_space)
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = classification_model[feasible_parameter].classes_.tolist().index(True)
        feasibility_indicator = classification_prediction_results[feasible_parameter][:,true_value_index]
    else:
        feasibility_indicator = [1]*number_of_predictions # if no classification model is used, then all points are feasible

    # Normalize predictions to [0, 1]
    normalized_predictions = {}
    for objective in regression_prediction_results:
        tmp_min = min(regression_prediction_results[objective])
        tmp_objective_limits[objective][0] = min(tmp_min, tmp_objective_limits[objective][0])
        tmp_max = max(regression_prediction_results[objective])
        tmp_objective_limits[objective][1] = max(tmp_max, tmp_objective_limits[objective][1])
        # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
        # We cannot optimize an objective when all values are the same, so we set it to 0
        if objective_limits[objective][1] == objective_limits[objective][0]:
            normalized_predictions[objective] = [0]*len(regression_prediction_results[objective])
        else:
            normalized_predictions[objective] = (regression_prediction_results[objective] - tmp_objective_limits[objective][0]) \
                                            /(tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0])

    total_weight = 0
    normalized_weights = {}
    if objective_bounds is not None: # Normalize weights to [0, 1] and sum(weights) = 1
        for objective in objective_weights:
            if tmp_objective_limits[objective][1] == tmp_objective_limits[objective][0]:
                normalized_weights[objective] = 0
            else:
                normalized_weights[objective] = (objective_weights[objective] - tmp_objective_limits[objective][0]) \
                                                /(tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0])
            total_weight += normalized_weights[objective]
        if total_weight == 0:
            total_weight = 1

        for objective in normalized_weights:
            normalized_weights[objective] = normalized_weights[objective]/total_weight
    else:
        normalized_weights = copy.deepcopy(objective_weights)

    # Compute scalarization
    if (scalarization_method == "linear"):
        scalarized_predictions = np.zeros(number_of_predictions)
        for run_index in range(number_of_predictions):
            for objective in regression_models:
                scalarized_predictions[run_index] += normalized_weights[objective] * normalized_predictions[objective][run_index]
            scalarized_predictions[run_index] = scalarized_predictions[run_index]*feasibility_indicator[run_index]
    # The paper does not propose this, I applied their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif(scalarization_method == "tchebyshev"):
        scalarized_predictions = scalarized_predictions = np.zeros(number_of_predictions)
        for run_index in range(number_of_predictions):
            total_value = 0
            for objective in regression_models:
                scalarized_value = normalized_weights[objective] * abs(normalized_predictions[objective][run_index])
                scalarized_predictions[run_index] = max(scalarized_value, scalarized_predictions[run_index])
                total_value += scalarized_value
            scalarized_predictions[run_index] += 0.05*total_value
            scalarized_predictions[run_index] = scalarized_predictions[run_index]*feasibility_indicator[run_index]
    elif(scalarization_method == "modified_tchebyshev"):
        scalarized_predictions = np.full((number_of_predictions), float("inf"))
        reciprocated_weights = reciprocate_weights(normalized_weights)
        for run_index in range(number_of_predictions):
            for objective in regression_models:
                scalarized_value = reciprocated_weights[objective] * abs(normalized_predictions[objective][run_index])
                scalarized_predictions[run_index] = min(scalarized_value, scalarized_predictions[run_index])
            scalarized_predictions[run_index] = scalarized_predictions[run_index]*feasibility_indicator[run_index]
            scalarized_predictions[run_index] = -scalarized_predictions[run_index] # We will minimize later, but we want to maximize instead, so we invert the sign
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
    objective_bounds,
    objective_limits,
    iteration_number,
    classification_model=None,
    tree_means_per_leaf=None,
    tree_vars_per_leaf=None,
    number_of_cpus=0):
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
    :param objective_bounds: a list containing user-defined bounds for the objectives bounding box.
    :param objective_limits: a dictionary with estimated minimum and maximum values for each objective.
    :param iteration_number: an integer for the current iteration number, used to compute the beta
    :param classification_model: the surrogate model used to evaluate feasibility constraints
    :param tree_means_per_leaf: prediction mean for each leaf of each tree.
    :param tree_vars_per_leaf: prediction variance for each leaf of each tree.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: a list of scalarized values for each point in bufferx.
    """
    augmentation_constant = 0.05
    prediction_means = {}
    prediction_variances = {}
    number_of_predictions = len(bufferx)
    tmp_objective_limits = copy.deepcopy(objective_limits)
    tree_predictions = models.tree_predictions(bufferx, regression_models, param_space)
    leaf_per_sample = models.get_leaves_per_sample(bufferx, regression_models, param_space)
    for objective in regression_models:
        tmp_min = np.amin(tree_predictions[objective])
        tmp_objective_limits[objective][0] = min(tmp_min, tmp_objective_limits[objective][0])
        tmp_max = np.amax(tree_predictions[objective])
        tmp_objective_limits[objective][1] = max(tmp_max, tmp_objective_limits[objective][1])

        prediction_means[objective] = models.compute_rf_prediction(leaf_per_sample[objective], tree_means_per_leaf[objective])
        prediction_variances[objective] = models.compute_rf_prediction_variance(
                                                                    leaf_per_sample[objective],
                                                                    prediction_means[objective],
                                                                    tree_means_per_leaf[objective],
                                                                    tree_vars_per_leaf[objective])
    # Normalize weights to [0, 1] and sum(weights) = 1
    total_weight = 0
    normalized_weights = {}
    if objective_bounds is not None: 
        for objective in objective_weights:
            # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
            # We cannot optimize an objective when all values are the same, so we set its weight to 0.
            if tmp_objective_limits[objective][1] == tmp_objective_limits[objective][0]:
                normalized_weights[objective] = 0
            else:
                normalized_weights[objective] = (objective_weights[objective] - tmp_objective_limits[objective][0]) \
                                                /(tmp_objective_limits[objective][1] - tmp_objective_limits[objective][0])

            total_weight += normalized_weights[objective]
        if total_weight == 0:
            total_weight = 1

        for objective in normalized_weights:
            normalized_weights[objective] = normalized_weights[objective]/total_weight
    else:
        normalized_weights = copy.deepcopy(objective_weights)

    if classification_model != None:
        classification_prediction_results = models.model_probabilities(bufferx, classification_model, param_space)
        feasible_parameter = param_space.get_feasible_parameter()[0]
        true_value_index = classification_model[feasible_parameter].classes_.tolist().index(True)
        feasibility_indicator = classification_prediction_results[feasible_parameter][:,true_value_index]
    else:
        feasibility_indicator = [1]*number_of_predictions # if no classification model is used, then all points are feasible

    data_array_scalarization, tmp_objective_limits = compute_data_array_scalarization(
                                                                                    data_array,
                                                                                    objective_weights,
                                                                                    tmp_objective_limits,
                                                                                    objective_bounds,
                                                                                    scalarization_method)

    f_min = min(data_array_scalarization)
    f_min = 1 - f_min
    # Compute scalarization
    if (scalarization_method == "linear"):
        scalarized_predictions = np.zeros(number_of_predictions)
        for x_index in range(number_of_predictions):
            scalarized_value = 0
            for objective in regression_models:
                x_var = prediction_variances[objective][x_index]
                x_std = np.sqrt(x_var)
                x_mean = prediction_means[objective][x_index]
                x_mean = 1 - x_mean
                v = (x_mean - f_min)/x_std
                objective_ei = (x_mean - f_min)*stats.norm.cdf(v) + x_std*stats.norm.pdf(v)
                scalarized_value += objective_ei*normalized_weights[objective]
            scalarized_predictions[x_index] = scalarized_value*feasibility_indicator[x_index]
            scalarized_predictions[x_index] = -1*scalarized_predictions[x_index]
    # The paper does not propose this, I applied their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif(scalarization_method == "tchebyshev"):
        scalarized_predictions = scalarized_predictions = np.zeros(number_of_predictions)
        for x_index in range(number_of_predictions):
            total_value = 0
            for objective in regression_models:
                x_var = prediction_variances[objective][x_index]
                x_std = np.sqrt(x_var)
                x_mean = prediction_means[objective][x_index]
                x_mean = 1 - x_mean
                v = (x_mean - f_min)/x_std
                objective_ei = (x_mean - f_min)*stats.norm.cdf(v) + x_std*stats.norm.pdf(v)
                scalarized_value = normalized_weights[objective] * objective_ei
                scalarized_predictions[x_index] = max(scalarized_value, scalarized_predictions[x_index])
                total_value += scalarized_value
            scalarized_predictions[x_index] += augmentation_constant*total_value
            scalarized_predictions[x_index] = scalarized_predictions[x_index]*feasibility_indicator[x_index]
            scalarized_predictions[x_index] = -1*scalarized_predictions[x_index]
    elif(scalarization_method == "modified_tchebyshev"):
        scalarized_predictions = np.full((number_of_predictions), float("inf"))
        reciprocated_weights = reciprocate_weights(normalized_weights)
        for x_index in range(number_of_predictions):
            for objective in regression_models:
                x_var = prediction_variances[objective][x_index]
                x_std = np.sqrt(x_var)
                x_mean = prediction_means[objective][x_index]
                x_mean = 1 - x_mean
                v = (x_mean - f_min)/x_std
                objective_ei = (x_mean - f_min)*stats.norm.cdf(v) + x_std*stats.norm.pdf(v)
                scalarized_value = reciprocated_weights[objective] * objective_ei
                scalarized_predictions[x_index] = min(scalarized_value, scalarized_predictions[x_index])
            scalarized_predictions[x_index] = scalarized_predictions[x_index]*feasibility_indicator[x_index]
    else:
        print("Error: unrecognized scalarization method:", scalarization_method)
        raise SystemExit

    return scalarized_predictions, tmp_objective_limits

def main(config, black_box_function=None, output_file=""):
    """
    Run design-space exploration using random scalarizations.
    :param config: dictionary containing all the configuration parameters of this design-space exploration.
    :param output_file: a name for the file used to save the dse results.
    :return:
    """
    start_time = (datetime.datetime.now())
    # unpack config into local variables
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
    input_params = param_space.get_input_parameters()
    number_of_objectives = len(optimization_metrics)
    optimization_iterations = config["optimization_iterations"]
    evaluations_per_optimization_iteration = config["evaluations_per_optimization_iteration"]
    number_of_cpus = config["number_of_cpus"]
    local_search_random_points = config["local_search_random_points"]
    scalarization_key = config["scalarization_key"]
    print_importances = config["print_parameter_importance"]
    epsilon_greedy_threshold = config["epsilon_greedy_threshold"]

    if "feasible_output" in config:
        feasible_output = config["feasible_output"]
        feasible_output_name = feasible_output["name"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
        enable_feasible_predictor_grid_search_on_recall_and_precision = feasible_output["enable_feasible_predictor_grid_search_on_recall_and_precision"]
        feasible_predictor_grid_search_validation_file = feasible_output["feasible_predictor_grid_search_validation_file"]
        feasible_parameter = param_space.get_feasible_parameter()

    acquisition_function = config["acquisition_function"]
    scalarization_method = config["scalarization_method"]
    local_search_starting_points = config["local_search_starting_points"]
    objective_bounds = None
    objective_limits = {}
    debug = False
    exhaustive_search_data_array = None
    tree_means_per_leaf=None,
    tree_vars_per_leaf=None,

    weight_sampling = config["weight_sampling"]
    if (weight_sampling == "bounding_box"):
        objective_bounds = {}
        user_bounds = config["bounding_box_limits"]
        if (len(user_bounds) == 2):
            if (user_bounds[0] > user_bounds[1]):
                user_bounds[0], user_bounds[1] = user_bounds[1], user_bounds[0]
            for objective in optimization_metrics:
                objective_bounds[objective] = user_bounds
                objective_limits[objective] = user_bounds
        elif (len(user_bounds) == number_of_objectives*2):
            idx = 0
            for objective in optimization_metrics:
                objective_bounds[objective] = user_bounds[idx:idx+2]
                if (objective_bounds[objective][0] > objective_bounds[objective][1]):
                    objective_bounds[objective][0], objective_bounds[objective][1] = objective_bounds[objective][1], objective_bounds[objective][0]
                objective_limits[objective] = objective_bounds[objective]
                idx += 2
        else:
            print("Wrong number of bounding boxes, expected 2 or", 2*number_of_objectives, "got", len(user_bounds))
            raise SystemExit
    else:
        for objective in optimization_metrics:
            objective_limits[objective] = [float("inf"), float("-inf")]

    doe_type = config["design_of_experiment"]["doe_type"]
    number_of_doe_samples = config["design_of_experiment"]["number_of_samples"]

    model = config["models"]["model"]
    if model == "random_forest":
        number_of_trees = config["models"]["number_of_trees"]

    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if (hypermapper_mode == 'client-server'):
        sys.stdout.switch_log_only_on_file(True)

    exhaustive_search_data_array = None
    exhaustive_search_fast_addressing_of_data_array = None
    if hypermapper_mode == 'exhaustive':
        exhaustive_file = config["hypermapper_mode"]["exhaustive_search_file"]
        exhaustive_search_data_array, exhaustive_search_fast_addressing_of_data_array = param_space.load_data_file(exhaustive_file, debug=False, number_of_cpus=number_of_cpus)

    if output_file == "":
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
    else:
        output_data_file = output_file

    beginning_of_time = param_space.current_milli_time()
    absolute_configuration_index = 0

    # Add default configuration to doe phase
    fast_addressing_of_data_array = {}
    default_configuration = param_space.get_default_or_random_configuration()
    str_data = param_space.get_unique_hash_string_from_values(default_configuration)
    fast_addressing_of_data_array[str_data] = absolute_configuration_index

    if number_of_doe_samples-1 > 0:
        configurations = param_space.get_doe_sample_configurations(fast_addressing_of_data_array, number_of_doe_samples-1, doe_type) + [default_configuration]
    else:
        configurations = [default_configuration]

    print("Design of experiment phase, number of doe samples = %d ......." % number_of_doe_samples)
    doe_t0 = datetime.datetime.now()
    data_array = param_space.run_configurations(
                                                hypermapper_mode,
                                                configurations,
                                                beginning_of_time,
                                                black_box_function,
                                                exhaustive_search_data_array,
                                                exhaustive_search_fast_addressing_of_data_array,
                                                run_directory)
    absolute_configuration_index += number_of_doe_samples
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

    for objective in optimization_metrics:
        lower_bound = min(objective_limits[objective][0], min(data_array[objective]))
        upper_bound = max(objective_limits[objective][1], max(data_array[objective]))
        objective_limits[objective] = [lower_bound, upper_bound]
    print("\nEnd of doe phase, the number of new configuration runs is: %d\n" %absolute_configuration_index)
    sys.stdout.write_to_logfile(("End of DoE - Time %10.4f sec\n" % ((datetime.datetime.now() - doe_t0).total_seconds())))
    if doe_type == "grid_search" and optimization_iterations > 0:
        print("Warning: DoE is grid search, setting number of optimization iterations to 0")
        optimization_iterations = 0

    # Main optimization loop
    if evaluations_per_optimization_iteration > 1:
        print("Warning, number of evaluations per iteration > 1")
        print("HyperMapper with random scalarizations currently does not support multiple runs per iteration, setting evaluations per iteration to 1")

    optimization_function_parameters = {}
    optimization_function_parameters['number_of_cpus'] = number_of_cpus
    optimization_function_parameters['param_space'] = param_space
    optimization_function_parameters['objective_bounds'] = objective_bounds
    optimization_function_parameters['scalarization_method'] = scalarization_method
    iteration_number = 0
    bo_t0 = datetime.datetime.now()
    while iteration_number < optimization_iterations:
        print("Starting optimization iteration", iteration_number+1)
        iteration_t0 = datetime.datetime.now()
        model_t0 = datetime.datetime.now()
        regression_models,_,_ = models.generate_mono_output_regression_models(
                                                                            data_array,
                                                                            param_space,
                                                                            input_params,
                                                                            optimization_metrics,
                                                                            1.00,
                                                                            n_estimators=number_of_trees,
                                                                            max_features=0.5,
                                                                            number_of_cpus=number_of_cpus,
                                                                            print_importances=print_importances)
        # If using ucb, get means and vars per leaf
        if (acquisition_function == "UCB") or (acquisition_function == "EI"):
            bufferx = [data_array[input_param] for input_param in input_params]
            bufferx = list(map(list, list(zip(*bufferx))))
            tree_means_per_leaf = {}
            tree_vars_per_leaf = {}
            leaf_per_sample = models.get_leaves_per_sample(bufferx, regression_models, param_space)
            for objective in optimization_metrics:
                # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
                # We cannot optimize an objective when all values are the same, so we set it to 0
                if objective_limits[objective][1] == objective_limits[objective][0]:
                    normalized_objective = [0]*len(data_array[objective])
                else:
                    normalized_objective = [(x - objective_limits[objective][0]) \
                                            /(objective_limits[objective][1] - objective_limits[objective][0]) for x in data_array[objective]]
                tree_means_per_leaf[objective] = models.get_mean_per_leaf(normalized_objective, leaf_per_sample[objective])
                tree_vars_per_leaf[objective] = models.get_var_per_leaf(normalized_objective, leaf_per_sample[objective])

        # Change splits of each node from (lower_bound + upper_bound)/2 to a uniformly sampled split in (lower_bound, upper_bound)
        regression_models = models.transform_rf_using_uniform_splits(regression_models, data_array, param_space)

        classification_model = None
        if enable_feasible_predictor:
            classification_model,_,_ = models.generate_classification_model(application_name,
                                                                            param_space,
                                                                            data_array,
                                                                            input_params,
                                                                            feasible_parameter,
                                                                            1.00,
                                                                            debug,
                                                                            n_estimators=number_of_trees,
                                                                            max_features=0.75,
                                                                            number_of_cpus=number_of_cpus,
                                                                            data_array_exhaustive=exhaustive_search_data_array,
                                                                            enable_feasible_predictor_grid_search_on_recall_and_precision=enable_feasible_predictor_grid_search_on_recall_and_precision,
                                                                            feasible_predictor_grid_search_validation_file=feasible_predictor_grid_search_validation_file,
                                                                            print_importances=print_importances)
        model_t1 = datetime.datetime.now()
        if (weight_sampling == "bounding_box"):
            objective_weights = sample_weight_bbox(optimization_metrics, objective_bounds, 1)[0]
        elif (weight_sampling == "flat"):
            objective_weights = sample_weight_flat(optimization_metrics, 1)[0]
        else:
            print("Error: unrecognized option:", weight_sampling)
            raise SystemExit

        epsilon = random.uniform(0,1)
        local_search_t0 = datetime.datetime.now()
        if epsilon > epsilon_greedy_threshold:
            optimization_function_parameters['acquisition_function'] = acquisition_function
            optimization_function_parameters['objective_weights'] = objective_weights
            optimization_function_parameters['regression_models'] = regression_models
            optimization_function_parameters['objective_limits'] = objective_limits
            optimization_function_parameters['iteration_number'] = iteration_number
            optimization_function_parameters['data_array'] = data_array
            optimization_function_parameters['classification_model'] = classification_model
            optimization_function_parameters['tree_means_per_leaf'] = tree_means_per_leaf
            optimization_function_parameters['tree_vars_per_leaf'] = tree_vars_per_leaf
            data_array_scalarization, objective_limits = compute_data_array_scalarization(
                                                                                        data_array,
                                                                                        objective_weights,
                                                                                        objective_limits,
                                                                                        objective_bounds,
                                                                                        scalarization_method)

            data_array[scalarization_key] = data_array_scalarization.tolist()
            _ , best_configuration = local_search(
                                                local_search_starting_points,
                                                local_search_random_points,
                                                param_space,
                                                fast_addressing_of_data_array,
                                                False, # we do not want the local search to consider feasibility constraints, only the acquisition functions
                                                run_acquisition_function,
                                                optimization_function_parameters,
                                                scalarization_key,
                                                previous_points=data_array)
        else:
            sys.stdout.write_to_logfile(str(epsilon) + " < " + str(epsilon_greedy_threshold) + " random sampling a configuration to run\n")
            best_configuration = param_space.random_sample_configurations_without_repetitions(fast_addressing_of_data_array, 1)[0]
        local_search_t1 = datetime.datetime.now()

        configurations = [best_configuration]
        str_data = param_space.get_unique_hash_string_from_values(best_configuration)
        fast_addressing_of_data_array[str_data] = absolute_configuration_index
        absolute_configuration_index += 1

        black_box_function_t0 = datetime.datetime.now()
        new_data_array = param_space.run_configurations(
                                                        hypermapper_mode,
                                                        configurations,
                                                        beginning_of_time,
                                                        black_box_function,
                                                        exhaustive_search_data_array,
                                                        exhaustive_search_fast_addressing_of_data_array,
                                                        run_directory)
        black_box_function_t1 = datetime.datetime.now()

        data_array = concatenate_data_dictionaries(
                                                new_data_array,
                                                data_array,
                                                param_space.input_output_and_timestamp_parameter_names)
        for objective in optimization_metrics:
            lower_bound = min(objective_limits[objective][0], min(data_array[objective]))
            upper_bound = max(objective_limits[objective][1], max(data_array[objective]))
            objective_limits[objective] = [lower_bound, upper_bound]

        iteration_number += 1
        sys.stdout.write_to_logfile(("Model fitting time %10.4f sec\n" % ((model_t1 - model_t0).total_seconds())))
        sys.stdout.write_to_logfile(("Local search time %10.4f sec\n" % ((local_search_t1 - local_search_t0).total_seconds())))
        sys.stdout.write_to_logfile(("Black box function time %10.4f sec\n" % ((black_box_function_t1 - black_box_function_t0).total_seconds())))
        sys.stdout.write_to_logfile(("Total iteration time %10.4f sec\n" % ((datetime.datetime.now() - iteration_t0).total_seconds())))
    sys.stdout.write_to_logfile(("End of BO phase - Time %10.4f sec\n" % ((datetime.datetime.now() - bo_t0).total_seconds())))

    with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'w') as f:
        w = csv.writer(f)
        w.writerow(list(data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, data_array) for j in list(data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in reversed(range(len(data_array[optimization_metrics[0]]))):
            w.writerow(tmp_list[i])


    print("End of Random Scalarizations")
    sys.stdout.write_to_logfile(("Total script time %10.2f sec\n" % ((datetime.datetime.now() - start_time).total_seconds())))

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
        os.chdir(hypermapper_pwd)
    except:
        pass

    print("### End of the random_scalarizations script.")
