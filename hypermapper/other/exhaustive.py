import itertools
import sys
import time

import torch

from hypermapper.param import space
from hypermapper.param.data import DataArray
from hypermapper.util.file import (
    initialize_output_data_file,
    load_previous,
)
from hypermapper.util.settings_check import settings_check_bo


def main(settings, black_box_function=None):
    """
    Run design-space exploration using bayesian optimization.
    Input:
        - settings: dictionary containing all the configuration parameters of this optimization.
        - black_box_function: the black box function to optimize (not needed in client-server mode).
    Returns:
        a DataArray object containing the data collected during the exhasutive search.
    """
    start_time = time.time()
    settings = settings_check_bo(settings, black_box_function)
    param_space = space.Space(settings)
    initialize_output_data_file(settings, param_space.all_names)

    data_array = DataArray(
        torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
    )

    beginning_of_time = param_space.current_milli_time()
    if settings["resume_optimization"]:
        data_array, absolute_configuration_index, beginning_of_time = load_previous(
            param_space, settings
        )
        space.write_data_array(param_space, data_array, settings["output_data_file"])

    if param_space.conditional_space:
        configurations = param_space.conditional_space_exhaustive_search()
    else:
        vals = []
        for param in param_space.parameters:
            vals.append(param.values)
        configurations = itertools.product(*vals)

    for configuration in configurations:
        str_data = param_space.get_unique_hash_string_from_values(configuration)
        if str_data in data_array.string_dict:
            configurations.remove(configuration)
    tmp_data_array = param_space.run_configurations(
        configurations, beginning_of_time, settings, black_box_function
    )
    data_array.cat(tmp_data_array)

    print("End of exhaustive search\n")

    sys.stdout.write_to_logfile(
        ("Total script time %10.2f sec\n" % (time.time() - start_time))
    )

    return data_array
