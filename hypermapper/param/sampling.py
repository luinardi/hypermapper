from typing import Optional, Dict, List

import numpy as np
import torch

from hypermapper.param import constraints
from hypermapper.param.space import Space
from hypermapper.param.parameters import Parameter


def random_sample(
        param_space: Space,
        n_samples: Optional[int] = 1,
        sampling_method: Optional[str] = "using_priors",
        allow_repetitions: Optional[bool] = False,
        previously_run: Dict[str, int] = None,
) -> torch.Tensor:
    """
    Random samples configurations
    Input:
        - param_space: Space object
        - n_samples: Number of samples to sample
        - sampling_method: "uniform", "using_priors", "embedding sample"
        - allow_repetitions: Allowing sampling duplicate samples. Generally much faster if allowed.
        - previously_run: tensor with previously run samples.
    Returns:
        - sampled configurations

    If repetitions are disallowed, it also ensures that previously_run_configurations is not among the returned configurations.
    If the space is conditional, it assumes the existence of a chain_of_trees.
    Will return n_samples new samples, not including the ones already in fast_address_of_data_array.
    Previously run is a dict with sting-hashes as keys for fast lookup.
    """
    if previously_run is None:
        previously_run = {}

    if not param_space.conditional_space:
        return _random_sample_non_constrained(
            param_space,
            n_samples,
            sampling_method,
            allow_repetitions,
            previously_run,
        ).to(dtype=torch.float64)

    else:
        return _random_sample_constrained(
            param_space,
            n_samples,
            sampling_method,
            allow_repetitions,
            previously_run,
        ).to(dtype=torch.float64)


def _random_sample_non_constrained(
        param_space: Space,
        n_samples: int,
        sampling_method: str,
        allow_repetitions: bool,
        previously_run: Dict[str, int],
) -> torch.Tensor:
    """
    Random samples configurations without constraints
    Input:
        - param_space: Space object
        - n_samples: Number of samples to sample
        - sampling_method: "uniform", "using_priors"
        - allow_repetitions: Allowing sampling duplicate samples. Generally much faster if allowed.
        - previously_run: tensor with previously run samples.
    Returns:
        - sampled configurations

    If repetitions are disallowed, it also ensures that previously_run_configurations is not among the returned configurations.
    If the space is conditional, it assumes the existence of a chain_of_trees.
    Will return n_samples new samples, not including the ones already in fast_adressing_of_data_array.
    """

    n_previously_run = len(list(previously_run.keys()))
    n_total_samples = n_samples + n_previously_run

    if allow_repetitions or param_space.has_real_parameters:
        parameters = param_space.parameters
        samples = torch.zeros((n_samples, len(parameters)))
        for i, parameter in enumerate(parameters):
            if sampling_method == "using_priors":
                samples[:, i] = parameter.sample(size=n_samples)
            else:
                samples[:, i] = parameter.sample(size=n_samples, uniform=True)
        return samples

    else:
        # depending on the size of the space, different approaches are more efficient
        if param_space.size <= n_total_samples:
            return param_space.filter_out_previously_run(param_space.get_space(), previously_run)

        elif param_space.size <= 2 * n_total_samples:
            remaining_space = param_space.filter_out_previously_run(param_space.get_space(), previously_run)
            remaining_space_size = param_space.size - n_previously_run

            if sampling_method == "uniform":
                probabilities = None

            elif sampling_method == "using_priors":
                # this uses the log-trick to improve numerical stability
                probabilities = np.array([
                    np.exp(np.sum([np.log(param.pdf(x)) for x, param in zip(remaining_space[configuration_idx], param_space.parameters)]))
                    for configuration_idx in range(remaining_space_size)
                ])
                probabilities = probabilities / np.sum(probabilities)
            else:
                raise Exception("Invalid sampling method", sampling_method)

            chosen_indices = np.random.choice(remaining_space_size, n_samples, replace=False, p=probabilities)

            return remaining_space[chosen_indices, :]

        else:
            """
            this part can probably be significantly improved.
            """

            configurations = torch.Tensor()
            parameters = param_space.parameters
            for trial in range(30):
                if configurations.shape[0] >= n_samples:
                    break

                new_configurations = torch.zeros((n_samples, len(parameters)))
                for i, parameter in enumerate(parameters):
                    if sampling_method == "using_priors":
                        new_configurations[:, i] = parameter.sample(size=n_samples)
                    else:
                        new_configurations[:, i] = parameter.sample(size=n_samples, uniform=True)

                new_configurations = param_space.filter_out_previously_run(new_configurations, previously_run)
                for c in new_configurations:
                    previously_run[param_space.get_unique_hash_string_from_values(c)] = -1
                configurations = torch.cat((configurations, new_configurations), dim=0)
            return configurations


def _random_sample_constrained(
        param_space: Space,
        n_samples: int,
        sampling_method: str,
        allow_repetitions: bool,
        previously_run_configurations: Dict[str, int],
):
    """
    Random samples constrained configurations
    Input:
        - param_space: Space object
        - n_samples: Number of samples to sample
        - sampling_method: "uniform", "using_priors", "embedding sample"
        - allow_repetitions: Allowing sampling duplicate samples. Generally much faster if allowed.
        - previously_run: tensor with previously run samples.
    Returns:
        - sampled configurations

    If repetitions are disallowed, it also ensures that previously_run_configurations is not among the returned configurations.
    If the space is conditional, it assumes the existence of a chain_of_trees.
    Will return "n_samples" new samples, not including the ones already in previously_run_configurations.
    """
    n_previously_run = len(list(previously_run_configurations.keys()))
    n_total_samples = n_samples + n_previously_run

    if allow_repetitions or param_space.has_real_parameters:
        tree_samples = param_space.chain_of_trees.sample(
            n_samples, sampling_method, [p.name for p in param_space.cot_parameters], allow_repetitions=True
        )
        non_tree_samples = _non_tree_constrained_sample(param_space, param_space.non_cot_parameters, n_samples, sampling_method)
        if tree_samples.shape[0] > 0 and non_tree_samples.shape[0] > 0:
            max_len = max(tree_samples.shape[0], non_tree_samples.shape[0])
            tree_samples = tree_samples[:max_len, :]
            non_tree_samples = non_tree_samples[:max_len, :]
        samples = torch.cat((tree_samples, non_tree_samples), dim=1)[:, param_space.cot_remap]
        return samples
    else:
        if param_space.chain_of_trees.get_size() <= n_total_samples:
            return param_space.filter_out_previously_run(
                param_space.chain_of_trees.get_all_configurations(),
                previously_run_configurations,
            )
        else:
            if not previously_run_configurations:
                return param_space.chain_of_trees.sample(
                    n_samples, sampling_method, param_space.parameter_names, allow_repetitions=False
                )
            else:
                too_many_samples = param_space.chain_of_trees.sample(
                    n_total_samples, sampling_method, param_space.parameter_names, allow_repetitions=False
                )
                return param_space.filter_out_previously_run(too_many_samples, previously_run_configurations)[:n_samples]


def _non_tree_constrained_sample(
    param_space: Space,
    parameters: List[Parameter],
    n_samples: int,
    sampling_method: str,
) -> torch.Tensor:
    """
    Sample constrained configurations by sampling from the prior and then filtering out the ones that are not feasible.
    used for configurations that are not in the chain of trees.
    Input:
        param_space: Space object
        parameters: List of parameters to sample
        n_samples: Number of samples to sample
        sampling_method: "uniform", "using_priors", "embedding sample"
    Returns:
        sampled configurations
    """

    samples = torch.Tensor()
    for trial in range(50):
        trial_samples = torch.zeros((3 * n_samples, len(parameters)))
        for i, parameter in enumerate(parameters):
            if sampling_method == "using_priors":
                trial_samples[:, i] = parameter.sample(size=3 * n_samples)
            else:
                trial_samples[:, i] = parameter.sample(size=3 * n_samples, uniform=True)
        transformed_configurations = {parameter.name: parameter_value for parameter, parameter_value in zip(parameters, [list(i) for i in zip(*param_space.convert(trial_samples, "internal", "original", parameters))])}
        feasible = constraints.evaluate_constraints(param_space.non_cot_constraints, transformed_configurations)
        samples = torch.cat((samples, trial_samples[feasible, :]), dim=0)
        if samples.shape[0] >= n_samples:
            break
    return samples[:n_samples, :]


def get_random_configurations(
        param_space: Space,
        use_priors=True,
        n_samples=1
) -> torch.Tensor:
    """
    Input:
        - param_space: Space object
        - use_priors: whether the prior distributions of the parameters should be used for the sampling
        - n_samples: the number of sampled random points
    Returns:
        - a number of random configurations from the parameter space under the form of a dictionary, or the sampled array, shape (size, dims)
    """

    parameters = param_space.parameters
    samples = torch.zeros((n_samples, len(parameters)))
    for i, parameter in enumerate(parameters):
        if use_priors:
            samples[i, :] = parameter.sample(size=n_samples)
        else:
            samples[i, :] = parameter.sample(size=n_samples, uniform=True)

    return samples
