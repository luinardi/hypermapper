import sys
from typing import List, Optional, Dict, Callable, Tuple

import torch
from scipy.stats import truncnorm

from hypermapper.param import space
from hypermapper.param.parameters import (
    Parameter,
    CategoricalParameter,
    PermutationParameter,
    OrdinalParameter,
    RealParameter,
    IntegerParameter,
)


def get_parameter_neighbors(
    configuration: torch.Tensor,
    parameter: Parameter,
    parameter_idx: int,
    only_smaller: Optional[bool] = False,
    only_larger: Optional[bool] = False,
    local_search_step_size: float = 0.03,
) -> torch.Tensor:
    """
    Returns neighbors for a single parameter.

    Input:
        - configuration: The configuration for which to find neighbors to.
        - parameter:
        - param_idx:
        - only_smaller: only returning smaller values for integers/ordinals (used in constrained spaces)
        - only_larger: only returning larger values for integers/ordinals (used in constrained spaces)
        - real_parameters: we only use local search for real parameters when random forests are involved
        - local_search_step_size: the standard deviation for the local search steps for real variables
    Returns:
        - Tensor of neighbors

    For categorical parameters, all neighbors are returned. For permutations all swap-1 neighbors. For ordinal the
    number is defined by the input arguments and for real and integer parameters, the sum of the two num_neighbors
    arguments is used but the distinction between above and below is ignored.

    Does not include the original configuration in neighbors.
    """

    neighbors = []
    if isinstance(parameter, CategoricalParameter):
        for value in parameter.get_discrete_values():
            if configuration[parameter_idx] != value:
                neighbor = configuration.clone()
                neighbor[parameter_idx] = value
                neighbors.append(neighbor)

    elif isinstance(parameter, PermutationParameter):
        if parameter.n_elements > 5:
            for i in range(parameter.n_elements):
                for j in range(i + 1, parameter.n_elements):
                    # swap i and j (probably sloooow)
                    neighbor = configuration.clone()
                    permutation: List[int] = list(
                        parameter.get_permutation_value(
                            int(configuration[parameter_idx].item())
                        )
                    )
                    j_val = permutation[j]
                    permutation[j] = permutation[i]
                    permutation[i] = j_val

                    neighbor[parameter_idx] = float(
                        parameter.get_int_value(tuple(permutation))
                    )
                    neighbors.append(neighbor)
        else:
            for value in parameter.get_discrete_values():
                if configuration[parameter_idx] != value:
                    neighbor = configuration.clone()
                    neighbor[parameter_idx] = float(value)
                    neighbors.append(neighbor)

    elif isinstance(parameter, OrdinalParameter):
        values = parameter.get_values()
        parameter_value = configuration[parameter_idx]
        value_idx = parameter.get_index(parameter_value.item())
        values_list = (
            [values[value_idx - 1]] if value_idx > 0 and not only_larger else []
        ) + (
            [values[value_idx + 1]]
            if value_idx < len(values) - 1 and not only_smaller
            else []
        )
        for value in values_list:
            neighbor = configuration.clone()
            neighbor[parameter_idx] = value
            neighbors.append(neighbor)

    elif isinstance(parameter, IntegerParameter):
        values_list = (
            [configuration[parameter_idx] - 1]
            if configuration[parameter_idx] > parameter.min_value and not only_larger
            else []
        ) + (
            [configuration[parameter_idx] + 1]
            if configuration[parameter_idx] < parameter.max_value and not only_smaller
            else []
        )
        for value in values_list:
            neighbor = configuration.clone()
            neighbor[parameter_idx] = value
            neighbors.append(neighbor)

    elif isinstance(parameter, RealParameter):
        mean = parameter.convert(configuration[parameter_idx], "internal", "01")
        scale = local_search_step_size
        a, b = (0 - mean) / scale, (1 - mean) / scale
        neighboring_values = truncnorm.rvs(a, b, loc=mean, scale=scale, size=6)
        for value in neighboring_values:
            neighbor = configuration.clone()
            neighbor[parameter_idx] = parameter.convert(value, "01", "internal")
            neighbors.append(neighbor)
    else:
        print("Unsupported parameter type")
        raise SystemExit

    if neighbors:
        return torch.cat([n.unsqueeze(0) for n in neighbors], 0)
    else:
        return torch.Tensor()


def get_neighbors(
    configuration: torch.Tensor, param_space: space.Space
) -> torch.Tensor:
    """
    Get the neighbors of a configuration

    Input:
        - configuration: dictionary containing the configuration we will generate neighbors for.
        - param_space: a space object containing the search space.
    Returns:
        - a torch.Tensor all the neighbors of 'configuration'
    """

    if param_space.conditional_space:
        return _generate_conditional_neighbors(configuration, param_space)
    else:
        parameters = param_space.parameters
        neighbors = configuration.unsqueeze(0)

        for parameter_idx, parameter in enumerate(parameters):
            if (
                isinstance(parameter, RealParameter)
                and param_space.use_gradient_descent
            ):
                continue
            neighbors = torch.cat(
                (
                    neighbors,
                    get_parameter_neighbors(
                        configuration,
                        parameter,
                        parameter_idx,
                        local_search_step_size=param_space.settings[
                            "local_search_step_size"
                        ],
                    ),
                ),
                0,
            )
        return neighbors


def _generate_conditional_neighbors(
    configuration: torch.Tensor, param_space: space.Space
) -> torch.Tensor:
    """
    Support method to get_neighbours()
    Input:
        - configuration: configuration for which to find neighbours for
        - param_space: the parameter space object
    Returns:
        - tensor with neighbors
    """

    parameters = param_space.parameters
    neighbors = [configuration]

    for parameter_idx, parameter in enumerate(parameters):
        parameter_type = param_space.parameter_types[parameter_idx]

        if parameter_type in ("categorical", "permutation"):
            parameter_neighbors = get_parameter_neighbors(
                configuration, parameter, parameter_idx
            )
            feasible = param_space.evaluate(parameter_neighbors, True)
            neighbors.extend(
                [neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f]
            )

        elif parameter_type in ["integer", "ordinal"]:
            """
            find feasible above
            """
            tmp_configuration = configuration
            while True:
                parameter_neighbors = get_parameter_neighbors(
                    tmp_configuration, parameter, parameter_idx, only_larger=True
                )
                if len(parameter_neighbors) == 0:
                    break
                if param_space.evaluate(parameter_neighbors, True)[0]:
                    neighbors.append(parameter_neighbors[0])
                    break
                tmp_configuration = parameter_neighbors[0]

            """
            find feasible below
            """
            tmp_configuration = configuration
            while True:
                parameter_neighbors = get_parameter_neighbors(
                    tmp_configuration, parameter, parameter_idx, only_smaller=True
                )
                if len(parameter_neighbors) == 0:
                    break
                if param_space.evaluate(parameter_neighbors, True)[0]:
                    neighbors.append(parameter_neighbors[0])
                    break
                tmp_configuration = parameter_neighbors[0]

        elif parameter_type == "real":
            if param_space.use_gradient_descent:
                continue
            parameter_neighbors = get_parameter_neighbors(
                configuration,
                parameter,
                parameter_idx,
                local_search_step_size=param_space.settings["local_search_step_size"],
            )
            feasible = param_space.evaluate(parameter_neighbors, True)
            neighbors.extend(
                [neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f]
            )

    if neighbors:
        return torch.cat([n.unsqueeze(0) for n in neighbors], 0)
    else:
        return torch.Tensor()


def local_search(
    start_configuration: torch.Tensor,
    settings: Dict,
    param_space: space.Space,
    acquisition_function: Callable,
    acquisition_function_parameters: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize the acquisition function using a mix of random and local search.
    This algorithm random samples N points and then does a local search on the
    best points from the random search and the best points from previous iterations (if any).

    Input:
        - start_configuration: the configuration from which to start the local search.
        - settings: Hypermapper run settings.
        - param_space: a space object containing the search space.
        - acquisition_function: the function that will be optimized by the local search.
        - acquisition_function_parameters: a dictionary containing the parameters that will be passed to the acquisition function.
    Returns:
        - best point found by the local search and its value.
    """

    configuration = start_configuration
    current_best_value = acquisition_function(
        settings,
        param_space,
        X=configuration.unsqueeze(0),
        **acquisition_function_parameters,
    )

    sys.stdout.write_to_logfile(
        "Starting local search on configuration: "
        + f"<{' '.join(str(x.item()) for x in configuration)}>"
        + f" with acq. val. {current_best_value.item()}"
        + "\n"
    )
    while True:
        neighbors = get_neighbors(configuration, param_space)
        neighbor_values = acquisition_function(
            settings, param_space, X=neighbors, **acquisition_function_parameters
        )
        if neighbor_values.shape[0] == 0:
            sys.stdout.write_to_logfile(
                "No neighbours found: "
                + f"<{' '.join(str(x.item()) for x in configuration)}>"
                + "\n"
            )
            break

        new_best_value, best_idx = torch.max(neighbor_values, 0)
        new_best_value = new_best_value.unsqueeze(0)
        best_neighbor = neighbors[best_idx]
        sys.stdout.write_to_logfile(
            "Best neighbour: "
            + f"<{' '.join(str(x.item()) for x in best_neighbor)}>"
            + f" with acq. val. {new_best_value.item()}"
            + "\n"
        )

        if (
            torch.all(torch.eq(configuration, best_neighbor))
            or new_best_value
            <= current_best_value + settings["local_search_improvement_threshold"]
        ):
            acquisition_function_parameters["verbose"] = True
            acquisition_function(
                settings,
                param_space,
                X=configuration.unsqueeze(0),
                **acquisition_function_parameters,
            )
            acquisition_function_parameters["verbose"] = False
            sys.stdout.write_to_logfile("Local minimum found!\n")
            break
        else:
            configuration = best_neighbor
            current_best_value = new_best_value

    return configuration, current_best_value
