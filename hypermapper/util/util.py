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
    return objective_means, objective_stds


####################################################
# Data structure handling
####################################################


def are_configurations_equal(configuration1, configuration2):
    """
    Compare two configurations. They are considered equal if they hold the same values for all keys.

    Input:
         - configuration1: the first configuration in the comparison
         - configuration2: the second configuration in the comparison
    Returns:
        - boolean indicating if configurations are equal or not
    """
    for c1, c2 in zip(configuration1, configuration2):
        if c1 != c2:
            return False
    return True


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
    best_indices = torch.sort(data_array.metrics_array[0]).indices[
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
    get_min_configurations(feasible_data_array, number_of_configurations)


def lex_sort_unique(matrix: np.ndarray) -> List[bool]:
    """
    checks uniqueness by first sorting the array lexicographically and then comparing neighbors.
    returns a list of bools indicating which indices contain first seen unique values.

    Input:
        - matrix: an np matrix with cnofigurations for rows and parameters for columns
    Returns:
        - a list of bools indicating which indices contain first seen unique values.
    """
    order = np.lexsort(matrix.T)
    matrix = matrix[order]
    diff = np.diff(matrix, axis=0)
    is_unique = np.ones(len(matrix), "bool")
    is_unique[1:] = (diff != 0).any(axis=1)
    return is_unique[order]


def remove_duplicate_configs(
    configurations: Union[np.ndarray, Tuple[np.ndarray]],
    ignore_columns=None,
):
    """
    Removes the duplicates from the combined configurations configs, and lets the first configs keep the remaining
    configurations from the duplicates

    Input:
        - configurations: the configurations to be checked for duplicates - duplicates are checked across all configurations, with the first occurrence being kept
        - ignore_column: don't consider the entered columns when checking for duplicates

    Returns:
        - the configurations with duplicates removed
    """
    if isinstance(configurations, tuple):
        merged_configs = np.concatenate(configurations, axis=0)
        config_lengths = [len(c) for c in configurations]
        if ignore_columns is not None:
            merged_configs = np.delete(merged_configs, ignore_columns, axis=1)
        # _, unique_indices = np.unique(merged_configs, return_index=True, axis=0)
        unique_indices = np.arange(len(merged_configs))[lex_sort_unique(merged_configs)]

        split_unique_indices = []
        for config_length in config_lengths:
            split_unique_indices.append(
                [i for i in unique_indices if 0 <= i < config_length]
            )
            unique_indices -= config_length
        return [
            configurations[i][split_unique_indices[i]]
            for i in range(len(configurations))
        ]

    else:
        configs_copy = copy.copy(configurations)
        if ignore_columns is not None:
            configs_copy = np.delete(configs_copy, ignore_columns, axis=1)
        # _, unique_indices = np.unique(configs_copy, return_index=True, axis=0)
        unique_indices = np.arange(len(configs_copy))[lex_sort_unique(configs_copy)]
        return configurations[unique_indices]


####################################################
# Visualization
####################################################
def get_next_color():
    get_next_color.ccycle = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
        (0, 200, 0),
        (0, 0, 0),
        # get_next_color.ccycle = [(101, 153, 255), (0, 0, 0), (100, 100, 100), (150, 100, 150), (150, 150, 150),
        # (192, 192, 192), (255, 0, 0), (255, 153, 0), (199, 233, 180), (9, 112, 84),
        (0, 128, 0),
        (0, 0, 0),
        (199, 233, 180),
        (9, 112, 84),
        (170, 163, 57),
        (255, 251, 188),
        (230, 224, 123),
        (110, 104, 14),
        (49, 46, 0),
        (138, 162, 54),
        (234, 248, 183),
        (197, 220, 118),
        (84, 105, 14),
        (37, 47, 0),
        (122, 41, 106),
        (213, 157, 202),
        (165, 88, 150),
        (79, 10, 66),
        (35, 0, 29),
        (65, 182, 196),
        (34, 94, 168),
        (12, 44, 132),
        (79, 44, 115),
        (181, 156, 207),
        (122, 89, 156),
        (44, 15, 74),
        (18, 2, 33),
    ]
    get_next_color.color_count += 1
    if get_next_color.color_count > 33:
        return 0, 0, 0
    else:
        a, b, c = get_next_color.ccycle[get_next_color.color_count - 1]
        return float(a) / 255, float(b) / 255, float(c) / 255


get_next_color.color_count = 0


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
