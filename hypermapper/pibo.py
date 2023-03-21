##################################################################################################################################
# This script implements the Prior-guided Bayesian Optimization method, presented in: https://openreview.net/forum?id=MMAeCXIa89 #
##################################################################################################################################
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
    from hypermapper.prior_optimization import compute_probability_from_prior
    from hypermapper.random_scalarizations import run_acquisition_function
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
    from hypermapper.prior_optimization import compute_probability_from_prior
    from hypermapper.random_scalarizations import run_acquisition_function
    from hypermapper.utility_functions import (
        dict_list_to_matrix,
        deal_with_relative_and_absolute_path,
    )
    from hypermapper.cma_es import cma_es


def prior_weighted_acquisition_function(
    configurations,
    param_space,
    acquisition_function_wrapper,
    acquisition_function,
    objective_weights,
    objective_limits,
    iteration_number,
    prior_beta,
    regression_models,
    classification_model,
    data_array,
    model_type,
    scalarization_method,
    prior_floor=10**-6,
):
    objective = param_space.get_optimization_parameters()[0]
    # need to subtract by optimum to ensure scale-invariance when multiplying with prior
    acquisition_function_values, feasibility_indicators = acquisition_function_wrapper(
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
    )

    numpy_acq_function_values = np.array(acquisition_function_values)[:, np.newaxis]

    if (acquisition_function == "TS") or (acquisition_function == "UCB"):
        numpy_acq_function_values = np.min(numpy_acq_function_values, 0)
    probabilities = compute_probability_from_prior(
        configurations, param_space, objective_weights
    )

    nonzero_probabilities = np.array(probabilities)[:, np.newaxis] + prior_floor
    prior_values = np.power(nonzero_probabilities, (prior_beta / iteration_number))
    return (
        numpy_acq_function_values * prior_values
    ).flatten().tolist(), feasibility_indicators


def run_pibo(
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
    # everything that gets passed to the acquisition function
    function_parameters = {}
    function_parameters["param_space"] = param_space
    function_parameters["iteration_number"] = iteration_number
    function_parameters["regression_models"] = regression_models
    function_parameters["classification_model"] = classification_model
    function_parameters["objective_weights"] = objective_weights
    function_parameters["objective_limits"] = objective_limits
    function_parameters["scalarization_method"] = config["scalarization_method"]
    # GP/RF
    function_parameters["model_type"] = config["models"]["model"]
    function_parameters["acquisition_function"] = config["acquisition_function"]
    function_parameters["acquisition_function_wrapper"] = run_acquisition_function
    function_parameters["data_array"] = data_array

    # Set the default value for beta unless specified otherwise
    if config["prior_beta"] == -1:
        function_parameters["prior_beta"] = config["optimization_iterations"] * 0.1

    else:
        function_parameters["prior_beta"] = config["prior_beta"]

    function_parameters["prior_floor"] = config["prior_floor"]
    optimization_metrics = param_space.get_optimization_parameters()

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
            prior_weighted_acquisition_function,
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
            prior_weighted_acquisition_function,
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
