from typing import Dict, Callable, Union


def settings_check_bo(
    settings: Dict, black_box_function: Union[None, Callable]
) -> Dict:
    """
    Initial consistency checks for BO.

    Input:
        - settings: The settings dict for the run.
        - black_box_function: The black box function to optimize.
    Returns:
        - updated settings
    """

    # Check if some parameters are correctly defined
    if settings["hypermapper_mode"]["mode"] == "default":
        if black_box_function is None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    if (settings["models"]["model"] == "gaussian_process") and (
        settings["acquisition_function"] == "TS"
    ):
        print(
            "Error: The TS acquisition function with Gaussian Process models is still under implementation"
        )
        print("Using EI acquisition function instead")
        settings["acquisition_function"] = "EI"

    if not settings["hypermapper_mode"]["mode"] in [
        "default",
        "client-server",
        "stateless",
    ]:
        print("Unrecognized hypermapper mode:", settings["hypermapper_mode"])
        raise SystemExit

    return settings
