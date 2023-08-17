###############################################################################################################################
# This script implements the Prior-guided Bayesian Optimization method, presented in: https://arxiv.org/abs/2204.11051       #
###############################################################################################################################
from typing import List
import torch

from hypermapper.bo.acquisition_functions import ei
from hypermapper.param.space import Space


def get_prior(
    X: torch.Tensor,
    param_space: Space,
) -> torch.Tensor:
    """
    Compute the probability of configurations being good according to the prior.
    Input:
        - X: tensor with configurations for which to calculate probability
        - param_space: Parameter Space object

    Return:
        - tensor with the probability of each configuration being good/optimal according to the prior.

    (I removed what seems to be a not working attempt to combine priors with MO.)
    """
    probabilities = torch.zeros(X.shape[0], dtype=torch.float64)

    for col in range(X.shape[1]):
        probabilities += torch.log(param_space.parameters[col].pdf(X[:, col]) + 1e-12)

    return torch.exp(probabilities)


def ei_pibo(
    settings: dict,
    param_space: Space,
    X: torch.Tensor,
    objective_weights: List[float],
    iteration_number: int,
    **kwargs,
) -> torch.Tensor:
    """
    Compute a (multi-objective) EI acquisition function on X.

    Input:
        - settings: run settings for hypermapper
        - param_space: the Space object
        - X: a list of tuples containing the points to predict and scalarize.
        - objective_weights: a list containing the weights for each objective.
        - iteration_number: an integer for the current iteration number, used to compute the beta
        - kwargs: used to pass all the additional parameters to EI
    Returns:
        - a tensor of scalarized values for each point in X.
    """

    acquisition_function_values = ei(
        settings,
        param_space,
        X,
        objective_weights,
        **kwargs,
    )

    prior = get_prior(X, param_space)
    pibo_multipliers = (prior + 1e-6) ** (settings["model_weight"] / iteration_number)
    return acquisition_function_values * pibo_multipliers
