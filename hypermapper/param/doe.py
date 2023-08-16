from typing import Optional

from hypermapper.param.data import DataArray
from hypermapper.param.sampling import random_sample
from hypermapper.param.space import Space


def get_doe_sample_configurations(
        param_space: Space,
        data_array: DataArray,
        n_samples: int,
        doe_type: str,
        allow_repetitions: Optional[bool] = False,
):
    """
    Get a list of n_samples configurations with no repetitions and that are not already present in fast_addressing_of_data_array.
    The configurations are sampled following the design of experiments (DOE) in the doe input variable.

    Input:
         - param_space: the Space object
         - data_array: previous points
         - n_samples: the number of unique samples required
         - doe_type: type of design of experiments (DOE) chosen
         - allow_repetitions: allow repeated configurations
    Returns:
        - torch.tensor
    """
    if doe_type == "random sampling":
        configurations = random_sample(
            param_space,
            n_samples,
            "uniform",
            allow_repetitions,
            data_array.string_dict,
        )
    elif doe_type == "embedding random sampling":
        configurations = random_sample(
            param_space,
            n_samples,
            "embedding",
            allow_repetitions,
            data_array.string_dict,
        )
    else:
        print("Error: design of experiment sampling method not found. Exit.")
        exit(1)
    return configurations
