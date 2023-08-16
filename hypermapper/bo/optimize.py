import copy
import sys
import time
from typing import Dict, Callable

import numpy as np
import torch
from scipy.optimize import minimize

from hypermapper.bo.acquisition_functions import ei, ucb
from hypermapper.bo.local_search import local_search
from hypermapper.bo.prior_acquisition_functions import ei_pibo
from hypermapper.param import space
from hypermapper.param.data import DataArray
from hypermapper.param.parameters import RealParameter
from hypermapper.param.sampling import random_sample


def optimize_scipy(
        start_configuration: torch.Tensor,
        settings: Dict,
        param_space: space.Space,
        acquisition_function: Callable,
        acquisition_function_parameters: Dict,
        scaling_value: float = 1,
):
    """
    Optimize the acquisition function over the real parameters using gradient descent.
    Input:
        start_configuration: The configuration to start from
        settings: a run settings dict
        param_space: a space object
        acquisition_function: The acquisition function to optimize
        acquisition_function_parameters: The parameters to pass to the acquisition function
        scaling_value: For numerical stability, it is better to scale the acquisition function when it gets too small (reverse internal scaling)
    Returns:
        - the best configuration found and its acquisition function value.

    """
    if scaling_value <= 1e-18:
        scaling_value = 1
    real_parameters = param_space.real_parameters
    real_parameter_indices = [param_space.parameters.index(p) for p in real_parameters]
    non_real_parameter_indices = [i for i in range(param_space.dimension) if i not in real_parameter_indices]
    x0 = np.array(start_configuration)
    bounds = [(p.min_value, p.max_value) if isinstance(p, RealParameter) else (-np.inf, np.inf) for p in param_space.parameters]

    def f_wrapper(x: np.ndarray):
        x = torch.from_numpy(x).to(start_configuration).contiguous().requires_grad_(True)
        columns = list(x.unbind(dim=-1))
        for index in non_real_parameter_indices:
            columns[index] = torch.full_like(columns[index], x[index].item())
        x_fix = torch.stack(columns, dim=-1)
        aq_val = acquisition_function(
            settings,
            param_space,
            X=x_fix.unsqueeze(0),
            **acquisition_function_parameters
        )
        grad = torch.autograd.grad(aq_val, x)[0].contiguous().view(-1).cpu().detach().contiguous().double().clone().numpy()
        return -aq_val.squeeze().item() / scaling_value, -grad / scaling_value

    res = minimize(f_wrapper, args=(), x0=x0, bounds=bounds, jac=True, options={"gtol": 1e-8})
    configuration = start_configuration.clone()
    res_x_torch = torch.tensor(res.x)
    for idx in real_parameter_indices:
        configuration[idx] = res_x_torch[idx]
    return configuration, -torch.tensor(res.fun).unsqueeze(0) * scaling_value


def optimize_acq(
        settings: Dict,
        param_space: space.Space,
        data_array: DataArray,
        regression_models,
        iteration_number,
        objective_weights,
        objective_means,
        objective_stds,
        best_values,
        classification_model=None,
):
    """
    Run one iteration of bayesian optimization with random scalarizations.

    Input:
        - settings: dictionary containing all the settings of this optimization.
        - param_space: parameter space object for the current application.
        - data_array: a dictionary containing previously explored points and their function values.
        - regression_models: the surrogate models used to evaluate points.
        - iteration_number: the current iteration number.
        - objective_weights: objective weights for multi-objective optimization. Not implemented yet.
        - objective_means: estimated means for each objective.
        - objective_stds: estimated standard deviations for each objective.
        - best_values: best values found so far for each objective.
        - classification_model: feasibility classifier for constrained optimization.

    Returns:
        - best configuration.
    """

    ### SETUP
    acquisition_function_parameters = {"regression_models": regression_models,
                                       "iteration_number": iteration_number,
                                       "classification_model": classification_model,
                                       "objective_weights": objective_weights,
                                       "objective_means": objective_means,
                                       "objective_stds": objective_stds,
                                       "best_values": best_values,
                                       "feasibility_threshold": (
                                           np.random.choice(settings["feasible_output"]["feasibility_threshold"] * 0.1 * np.arange(11))
                                           if "feasible_output" in settings else 0)
                                       }

    if settings["acquisition_function"] == "EI":
        acquisition_function = ei

    elif settings["acquisition_function"] == "UCB":
        acquisition_function = ucb

    elif settings["acquisition_function"] == "EI_PIBO":
        acquisition_function = ei_pibo
    else:
        raise Exception("Invalid acquisition function specified.")

    t0 = time.time()

    ### INITIAL SAMPLES
    samples_from_prior = False
    if samples_from_prior:
        random_sample_configurations = torch.cat((
            random_sample(
                param_space,
                settings["local_search_random_points"],
                "uniform",
                False,
            ) + random_sample(
                param_space,
                settings["local_search_random_points"],
                "using_priors",
                False,
            )), 0
        )

    else:
        random_sample_configurations = random_sample(
            param_space,
            settings["local_search_random_points"],
            "uniform",
            False,
        )

    sys.stdout.write_to_logfile("Total RS time %10.4f sec\n" % (time.time() - t0))
    t1 = time.time()

    rs_acquisition_values = acquisition_function(
        settings,
        param_space,
        X=random_sample_configurations,
        **acquisition_function_parameters
    )

    sys.stdout.write_to_logfile("Optimization function time %10.4f sec\n" % (time.time() - t1))
    if settings["local_search_starting_points"] == 0:
        best_index = torch.argmax(rs_acquisition_values).item()
        return random_sample_configurations[best_index]

    ### LOCAL SEARCH
    n_start_points = min(settings["local_search_starting_points"], rs_acquisition_values.shape[0])
    start_values, start_indices = torch.topk(rs_acquisition_values, n_start_points)
    start_configurations = random_sample_configurations[start_indices, :]

    if settings["local_search_from_best"]:
        previous_configuration_indices = torch.topk(data_array.metrics_array, largest=False, k=3, dim=0).indices.reshape(-1)
        previous_configurations = data_array.parameters_array[previous_configuration_indices]
        previous_values = acquisition_function(
            settings,
            param_space,
            X=previous_configurations,
            **acquisition_function_parameters
        )
        start_configurations = torch.cat((start_configurations, previous_configurations), 0)
        start_values = torch.cat((start_values, previous_values), 0)

    best_configurations = torch.Tensor()
    best_values = torch.Tensor()
    ls_time_start = time.time()

    """
    This loop switches between gradient descent for the continuous variables and local search for the discrete variables.
    Each is run until convergence, it this goes on until both have converged. 
    """
    for start_configuration, start_value in zip(start_configurations, start_values):
        previous_value = -np.inf
        value = start_value.unsqueeze(0)
        configuration = start_configuration
        torch.set_printoptions(precision=20)
        while value > previous_value + 1e-12:
            # discrete
            previous_value = value
            if (not param_space.only_real_parameters) or (not param_space.use_gradient_descent):
                configuration, value = local_search(
                    start_configuration=configuration,
                    settings=settings,
                    param_space=param_space,
                    acquisition_function=acquisition_function,
                    acquisition_function_parameters=acquisition_function_parameters,
                )

            if not param_space.use_gradient_descent:
                break
            # continuous
            configuration, value = optimize_scipy(
                start_configuration=configuration,
                settings=settings,
                param_space=param_space,
                acquisition_function=acquisition_function,
                acquisition_function_parameters=acquisition_function_parameters,
                scaling_value=torch.max(start_values).item(),
            )
            if param_space.only_real_parameters:
                break
        best_configurations = torch.cat((best_configurations, configuration.unsqueeze(0)))
        best_values = torch.cat((best_values, value), 0)

    #######################
    # This is for testing and debugging purposes only
    ###
    try:
        normalized_start_configurations = copy.deepcopy(start_configurations)
        for i, p in enumerate(param_space.parameters):
            normalized_start_configurations[:, i] = (normalized_start_configurations[:, i] - p.min_value) / (p.max_value - p.min_value)
        start_conf_dists = torch.tensor([
            [torch.norm(normalized_start_configurations[i] - normalized_start_configurations[j]) for i in range(start_configurations.shape[0])]
            for j in range(start_configurations.shape[0])
        ]).numpy()
        start_val_dists = torch.tensor([
            [torch.abs(start_values[i] - start_values[j]) for i in range(start_values.shape[0])]
            for j in range(start_values.shape[0])
        ]).numpy()
        sys.stdout.write_to_logfile("\nLocal search statistics:\n________________________\n")
        sys.stdout.write_to_logfile(f"Best start value: {start_values.max()}\n")
        sys.stdout.write_to_logfile(f"Worst start value: {start_values.min()}\n")
        sys.stdout.write_to_logfile(f"Best gd value: {best_values.max()}\n")
        sys.stdout.write_to_logfile(f"Worst gd value: {best_values.min()}\n")
        sys.stdout.write_to_logfile(f"x diffs: {start_conf_dists[np.nonzero(start_conf_dists)].min()}-{start_conf_dists.max()}\n")
        sys.stdout.write_to_logfile(f"y diffs: {start_val_dists[np.nonzero(start_val_dists)].min()}-{start_val_dists.max()}\n\n")
    except:
        pass
    ########################

    best_idx = torch.argmax(best_values)
    best_configuration = best_configurations[best_idx]

    sys.stdout.write_to_logfile("Multi-start LS time %10.4f sec\n" % (time.time() - ls_time_start))
    sys.stdout.write_to_logfile(
        "Best found configuration: "
        + f"<{' '.join(str(x.item()) for x in best_configuration)}>"
        + "\n"
    )
    acquisition_function_parameters["verbose"] = True
    acquisition_function(
        settings,
        param_space,
        X=best_configuration.unsqueeze(0),
        **acquisition_function_parameters,
    )
    acquisition_function_parameters["verbose"] = False

    return best_configuration
