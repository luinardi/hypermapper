from typing import Any, Dict, List, Union

import numexpr as ne
import numpy as np

from hypermapper.param.parameters import Parameter


def evaluate_constraints(
        constraints: List[str],
        configurations: Dict[str, List[Any]],
) -> List[bool]:
    """
    Checks configuration feasibility
    Input:
        - constraints: constraints to evaluate
        - configurations: configurations to evaluate ("original" representation)
    Returns:
        - List of booleans denoting whether the configurations are feasible

    Note that this requires some preprocessing - especially, the configurations should be provided as dicts.
    This preprocessing is performed by evaluate() in space.py.
    """

    # protect against empty configuration dicts
    if len(list(configurations.values())[0]) == 0:
        return []
    else:
        n_configurations = len(list(configurations.values())[0])

    # transform all permutation variables
    permutation_configurations = {}
    for varname in configurations:
        if type(configurations[varname][0]) in (list, tuple):
            for i in range(len(configurations[varname][0])):
                permutation_configurations[f"{varname}_{i}"] = [configurations[varname][j][i] for j in range(n_configurations)]
                permutation_configurations[f"{varname}_i{i}"] = [list(configurations[varname][j]).index(i) for j in range(n_configurations)]

    feasible = np.array([True for x in range(n_configurations)])
    for constraint in constraints:
        feasible = feasible & ne.evaluate(constraint, {**configurations, **permutation_configurations})
    return list(feasible)


def filter_conditional_values(
        parameter: Parameter,
        constraints: Union[List[str], None],
        partial_configuration: Dict[str, Any]
) -> List[Any]:
    """
    Returns all of its values which are feasible with regards to its constraints given previous values given in partial_configuration.
        Input:
            - parameter: parameter to find feasible values for
            - constraints: constraints to evaluate
            - partial_configuration: configuration so far ("original" representation)
        Returns:
            - List of feasible parameter values ("internal" representation)
    """
    if constraints is None:
        return parameter.values
    configurations = {
        **{kv[0]: [kv[1]] * len(parameter.values) for kv in partial_configuration.items()},
        **{parameter.name: [parameter.convert(v, "internal", "original") for v in parameter.values]}
    }
    feasible = evaluate_constraints(constraints, configurations)
    return [value for idx, value in enumerate(parameter.values) if feasible[idx]]
