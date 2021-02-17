import os
import sys
import warnings
from collections import OrderedDict

# ensure backward compatibility
# backward compatibility, check if imported as module
if __name__ != "__main__":
    warnings.warn(
        "\n\t*****\n\tHyperMapper seems to have been imported as a module.\n\tThis might lead to errors.\n\t"
        "Please be sure that you want to do this.\n\t"
        "Otherwise, this is probably caused by a misconfiguration of HyperMapper when trying to execute an "
        "example.\n\t"
        "Please choose one of the following options to fix this issue.\n\t"
        "1) (recommended)\n\t\t* Remove '.../hypermapper/scripts' from your PYTHONPATH.\n\t\t"
        "* Run your example from the HyperMapper root directory "
        "with 'python -m dir1.dir2.yourexample'\n\t2)\n\t\t"
        "* Update your PYTHONPATH from '.../hypermapper/scripts' to '.../hypermapper'.\n\t\t"
        "* Run your script as before.\n\t*****"
    )
try:
    from hypermapper import optimizer
except ImportError:
    if os.getenv("HYPERMAPPER_HOME"):  # noqa
        warnings.warn(
            "Found environment variable 'HYPERMAPPER_HOME', used to update the system path. Support might be discontinued in the future. Please make sure your installation is working without this environment variable, e.g., by installing with 'pip install hypermapper'.",
            DeprecationWarning,
            2,
        )  # noqa
        sys.path.append(os.environ["HYPERMAPPER_HOME"])  # noqa

    scripts_path = ["hypermapper/scripts", "hypermapper_dev/scripts"]

    if os.getenv("HYPERMAPPER_HOME"):
        scripts_path.append(os.path.join(os.getenv("HYPERMAPPER_HOME"), "scripts"))

    truncated_items = [
        p for p in sys.path if len([q for q in scripts_path if q in p]) == 0
    ]
    if len(truncated_items) < len(sys.path):
        warnings.warn(
            "Found hypermapper in PYTHONPATH. Usage is deprecated and might break things. "
            "Please remove all hypermapper references from PYTHONPATH. Trying to import "
            "without hypermapper in PYTHONPATH..."
        )
        sys.path = truncated_items
    else:
        # this removes the 'scripts' path from the sys path, enabling from importing from the hypermapper directory (name clash)
        # only necessary in 'scripts' directory, all imports from scripts have to be done above
        sys.path = (
            sys.path[1:]
            if "hypermapper/scripts" in sys.path[0]
            or "hypermapper_dev/scripts" in sys.path[0]
            else sys.path
        )
    sys.path.append(".")  # noqa
    sys.path = list(OrderedDict.fromkeys(sys.path))

    from hypermapper import optimizer


if __name__ == "__main__":
    warnings.warn(
        "Using 'scripts/hypermapper' is deprecated and it will be removed in the future. Use 'hypermapper/optimizer' instead.",
        DeprecationWarning,
        2,
    )
    if len(sys.argv) == 2:
        parameters_file = sys.argv[1]
    else:
        print("Error: only one argument needed, the parameters json file.")

    if parameters_file == "--help" or len(sys.argv) != 2:
        print("################################################")
        print("### Example: ")
        print("### cd hypermapper")
        print(
            "### python3 hypermapper/optimizer.py example_scenarios/spatial/BlackScholes_scenario.json"
        )
        print("################################################")
        exit(1)

    optimizer.optimize(parameters_file)
