import copy
from typing import List, Union, Tuple, Dict

import numpy as np
import torch
from scipy import stats

from hypermapper.param.data import DataArray


def update_mean_std(values: torch.Tensor, settings: Dict):
    """
    Update the mean and standard deviation of the objective function values.
    Args:
        values: the objective function values
        settings: the settings of the optimization

    Returns:
        the updated mean and standard deviation of the objective function values
    """

    if settings["log_transform_output"]:
        if torch.min(values) < 0:
            raise Exception("Can't log transform data that take negative values")
        objective_means = torch.mean(torch.log10(values), 0)
        objective_stds = torch.std(torch.log10(values), 0)
    else:
        objective_means = torch.mean(values, 0)
        objective_stds = torch.std(values, 0)
    if torch.any(objective_stds == 0):
        objective_stds[objective_stds == 0] = 1
    return objective_means, objective_stds


####################################################
# Data structure handling
####################################################

def get_min_configurations(
    data_array: DataArray, number_of_configurations: int
) -> DataArray:
    """
    Get the configurations with minimum value according to the comparison key

    Input:
         - configurations: dictionary containing the configurations.
         - number_of_configurations: number of configurations to return.
    Returns:
        - A DataArray with the best points
    """

    if data_array.metrics_array.shape[1] > 1:
        raise Exception(
            "Calling min config on a multi-objective problem which is not implemented. "
        )

    number_of_configurations = min(
        number_of_configurations, data_array.metrics_array.shape[0]
    )
    best_indices = torch.sort(data_array.metrics_array[:, 0]).indices[
        :number_of_configurations
    ]

    return data_array.slice(best_indices)


def get_min_feasible_configurations(
    data_array: DataArray, number_of_configurations: int
):
    """
    Input:
         - data_array: The data among which to select the points
         - number_of_configurations: number of configurations to return.
    Returns:
        - a dictionary containing the best configurations.
    """
    feasible_data_array = data_array.get_feasible()
    return get_min_configurations(feasible_data_array, number_of_configurations)


####################################################
# Scalarization
####################################################

def sample_weight_flat(optimization_metrics):
    """
    Sample lambdas for each objective following a dirichlet distribution with alphas equal to 1.
    In practice, this means we sample the weights uniformly from the set of possible weight vectors.
    Input:
         - optimization_metrics: a list containing the optimization objectives.
    Returns:
        - a list containing the weight of each objective.
    """
    alphas = np.ones(len(optimization_metrics))
    sampled_weights = stats.dirichlet.rvs(alpha=alphas, size=1)

    return torch.tensor(sampled_weights)
