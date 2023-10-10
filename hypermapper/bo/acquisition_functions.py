import sys
from typing import Dict, List, Any

import numpy as np
import torch

from hypermapper.bo.models import models
from hypermapper.param.space import Space


def ucb(
    settings: Dict,
    param_space: Space,
    X: torch.Tensor,
    objective_weights: torch.Tensor,
    regression_models: List[Any],
    iteration_number: int,
    classification_model: Any,
    feasibility_threshold,
    **kwargs,
) -> torch.Tensor:
    """
    Multi-objective ucb acquisition function as detailed in https://arxiv.org/abs/1805.12168.
    The mean and variance of the predictions are computed as defined by Hutter et al.: https://arxiv.org/pdf/1211.0906.pdf

    Input:
        - settings: the Hypermapper run settings
        - param_space: a space object containing the search space.
        - X: a list of tuples containing the points to predict and scalarize.
        - objective_weights: a list containing the weights for each objective.
        - regression_models: the surrogate models used to evaluate points.
        - iteration_number: an integer for the current iteration number, used to compute the beta
        - classification_model: the surrogate model used to evaluate feasibility constraints
        - feasibility_threshold: minimum probability of feasibility
        - kwargs: to throw away additional input
    Returns:
        - a tensor of scalarized values for each point in X.
    """
    beta = np.sqrt(0.125 * np.log(2 * iteration_number + 1))
    number_of_predictions = X.shape[0]

    prediction_means, prediction_stds = models.compute_model_mean_and_uncertainty(
        X,
        regression_models,
        param_space,
        predict_noiseless=settings["predict_noiseless"],
    )

    if classification_model is not None:
        feasibility_indicator = classification_model.feas_probability(X)
    else:
        feasibility_indicator = torch.ones(number_of_predictions)

    acq_val = (
        (prediction_means + beta * prediction_stds)
        @ objective_weights
        * feasibility_indicator
        * (feasibility_indicator >= feasibility_threshold)
    )

    return acq_val


def ei(
    settings: dict,
    param_space: Space,
    X: torch.Tensor,
    objective_weights: List[float],
    regression_models: List[Any],
    best_values: float,
    objective_means: torch.Tensor,
    objective_stds: torch.Tensor,
    classification_model: Any,
    feasibility_threshold: float,
    verbose: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Compute a (multi-objective) EI acquisition function on X.

    Input:
        - settings: run settings for hypermapper
        - param_space: the Space object
        - X: a list of tuples containing the points to predict and scalarize.
        - objective_weights: a list containing the weights for each objective.
        - regression_models: the surrogate models used to evaluate points.
        - best_values: best values observed until this point for each objective
        - objective_means: a dictionary with estimated (log) mean values for each objective.
        - objective_stds: a dictionary with estimated (log) std values for each objective.
        - classification_model: the surrogate model used to evaluate feasibility constraints
        - feasibility_threshold: minimum probability of feasibility
        - verbose: whether to print to the log file
        - kwargs: to throw away additional input
    Returns:
        - a tensor of scalarized values for each point in X.
    """
    number_of_predictions = X.shape[0]
    prediction_means, prediction_stds = models.compute_model_mean_and_uncertainty(
        X,
        regression_models,
        param_space,
        predict_noiseless=settings["predict_noiseless"],
    )

    if classification_model is not None:
        feasibility_indicator = classification_model.feas_probability(X)
    else:
        feasibility_indicator = torch.ones(number_of_predictions)

    xi = settings["exploration_augmentation"]

    if settings["log_transform_output"]:
        best_values = np.log10(best_values)

    if settings["objective_value_target"] != -9999:
        best_values = min(best_values, settings["objective_value_target"])

    normalized_best_values = (best_values - objective_means) / objective_stds
    f_stds = prediction_stds
    f_means = prediction_means
    v = (normalized_best_values - f_means - xi) / f_stds
    normal = torch.distributions.Normal(torch.zeros_like(v), torch.ones_like(v))
    try:
        objective_ei = (normalized_best_values - f_means - xi) * normal.cdf(
            v
        ) + f_stds * torch.exp(normal.log_prob(v))
    except Exception as e:
        print("SOME DEBUG INFO")
        print("v", v)
        print(normalized_best_values)
        print(f_means)
        print(objective_stds)
        print(f_stds)
        raise e
    scalarized_ei = objective_ei @ objective_weights
    if verbose and objective_ei.shape[0] == 1:
        sys.stdout.write_to_logfile(
            f"obj mean:{objective_means.squeeze().item()}  "
            + f"obj std:{objective_stds.squeeze().item()} \n"
        )
        sys.stdout.write_to_logfile(
            f"f*:{' '.join(str(x.item()) for x in normalized_best_values)}  "
            + f"mu:{' '.join(str(x.item()) for x in f_means.squeeze(0))}  "
            + f"sigma:{' '.join(str(x.item()) for x in f_stds.squeeze(0))}  "
            + f"feasibility:{feasibility_indicator.item()}  "
            + f"p1:{' '.join(str(x.item()) for x in ((normalized_best_values - f_means - xi) * normal.cdf(v)).squeeze(0))} "
            + f"p2:{' '.join(str(x.item()) for x in (f_stds * torch.exp(normal.log_prob(v))).squeeze(0))}  "
            + f"alpha:{' '.join(str(x.item()) for x in objective_ei.squeeze(0))}  "
            + f"w:{' '.join(str(x) for x in objective_weights)}\n"
        )
        sys.stdout.write_to_logfile(
            f"log EI value: {torch.log10(scalarized_ei).item()}\n"
        )
    if settings["log_acq_value"]:
        acq_val = torch.log10(
            scalarized_ei
            * feasibility_indicator
            * (feasibility_indicator >= feasibility_threshold)
            + 1e-18
        )
    else:
        acq_val = (
            scalarized_ei
            * feasibility_indicator
            * (feasibility_indicator >= feasibility_threshold)
        )
    return acq_val
