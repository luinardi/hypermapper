import sys
from typing import Dict

from sklearn.preprocessing import OneHotEncoder

from hypermapper.param.data import DataArray
from hypermapper.param.parameters import *
from hypermapper.param.space import Space


def transform_data(
        settings: Dict,
        data_array: DataArray,
        param_space: Space,
        objective_means: torch.Tensor,
        objective_stds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Transform the data array into a format that can be used by the GP models. It normalizes and the input, standardizes the output
    performs one-hot encoding of the categorical variables, parameterizes the permutation variables and log-transforms input and output if requested.
    Input:
        - settings: the settings dictionary
        - data_array: the data array to transform
        - param_space: the Space object
        - objective_means: the means of the objectives
        - objective_stds: the standard deviations of the objectives
    Returns:
        - the transformed input
        - the transformed output
        - the names of the paramterized parameters

    """
    # Transform input
    X, parametrization_names = preprocess_parameters_array(data_array.parameters_array, param_space)
    # Transform output
    Y = data_array.metrics_array.clone()
    if settings["log_transform_output"]:
        Y = torch.log10(Y)

    if settings["standardize_objectives"]:
        if not (objective_means is None or objective_stds is None):
            Y = (Y - torch.ones(Y.shape) * objective_means) / (torch.ones(Y.shape) * objective_stds)
        else:
            sys.stdout.write_to_logfile(
                "Warning: no statistics provided, skipping objective standardization.\n"
            )

    return X, Y, parametrization_names


def transform_estimate(
        settings: Dict,
        std_estimate: torch.Tensor,
        objective_means: torch.Tensor,
        objective_stds: torch.Tensor,
):
    """
    Transform the estimate of the standard deviation of the noise used in fixed and heteroskedastic GP noise models.
    Input:
        - settings: the settings dictionary
        - std_estimate: the estimate of the standard deviation
        - objective_means: the means of the objectives
        - objective_stds: the standard deviations of the objectives
    """

    if settings["log_transform_output"]:
        raise Exception("Log transform output not supported with Fixed/Heteroskedastic noise.")
    if settings["standardize_objectives"] and not (objective_means is None or objective_stds is None):
        std_estimate = std_estimate / objective_stds
    return std_estimate


def preprocess_parameters_array(
        X: torch.Tensor,
        param_space: Space,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Preprocess a data_array before feeding into a regression/classification model.
    The preprocessing standardize non-categorical inputs (if the flag is set).
    It also transforms categorical variables using one-hot encoding and permutation variables
    according to their chosen parametrization.

    Input:
         - data_array: DataArray containing the X values to transform
         - param_space: parameter space object for the current application.
    Returns:
        - Preprocessed X values
        - List of names of the parametrized parameters
    """
    X = X.clone()
    new_X = torch.Tensor()
    new_names = []
    for idx, parameter in enumerate(param_space.parameters):
        if (
                isinstance(parameter, RealParameter) or
                isinstance(parameter, IntegerParameter) or
                isinstance(parameter, OrdinalParameter)
        ):
            new_names.append(parameter.name)
            new_X = torch.cat((new_X, X[:, idx].unsqueeze(1)), dim=1)
            if parameter.transform == "log":
                new_X[:, -1] = torch.log10(new_X[:, -1])
            if param_space.normalize_inputs:
                p_min = parameter.get_min()
                p_max = parameter.get_max()
                if parameter.transform == "log":
                    p_min = np.log10(p_min)
                    p_max = np.log10(p_max)
                new_X[:, -1] = (new_X[:, -1] - p_min) / (p_max - p_min)
        elif isinstance(parameter, CategoricalParameter):
            # Categorical variables are encoded as their index, generate a list of "index labels"
            categories = np.arange(parameter.get_size())
            encoder = OneHotEncoder(categories="auto", sparse=False)
            encoder.fit(categories.reshape(-1, 1))
            x = np.array(X[:, idx]).reshape(-1, 1)
            encoded_x = encoder.transform(x)
            for i in range(encoded_x.shape[1]):
                new_names.append(f"{parameter.name}_{categories[i]}")
            new_X = torch.cat((new_X, torch.tensor(encoded_x)), dim=1)

        elif isinstance(parameter, PermutationParameter):
            # Permutation variables are encoded based on their chosen parametrization
            keys, encoded_x = parameter.parametrize(X[:, idx])
            new_names.extend(keys)
            new_X = torch.cat((new_X, torch.tensor(encoded_x)), dim=1)
    return new_X, new_names
