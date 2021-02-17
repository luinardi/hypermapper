import warnings
import sys
import os
from collections import OrderedDict

# ensure backward compatibility
try:
    from hypermapper.compute_pareto import compute
    from hypermapper.utility_functions import Logger
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
            "Please remove all hypermapper references from PYTHONPATH. Trying to import"
            "without hypermapper in PYTHONPATH..."
        )
        sys.path = truncated_items
    else:
        # this removes the 'scripts' path from the sys path, enabling from importing from the hypermapper directory (name clash)
        # only necessary in 'scripts' directory, all imports from scripts have to be done above
        sys.path = sys.path[1:]

    sys.path.append(".")  # noqa
    sys.path = list(OrderedDict.fromkeys(sys.path))

    from hypermapper.compute_pareto import compute
    from hypermapper.utility_functions import Logger

if __name__ == "__main__":
    warnings.warn(
        "Using 'scripts/compute_pareto' is deprecated and it will be removed in the future. Use 'hypermapper/compute_pareto' instead.",
        DeprecationWarning,
        2,
    )
    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the interactive mode we only want to log on the file.
    sys.stdout = Logger()

    parameters_file = ""
    input_data_file = None
    output_pareto_file = None
    if len(sys.argv) >= 2:
        parameters_file = sys.argv[1]
        if len(sys.argv) >= 3:
            input_data_file = sys.argv[2]
            if len(sys.argv) >= 4:
                output_pareto_file = sys.argv[3]
            if len(sys.argv) >= 5:
                print("Error: too many arguments.")
    else:
        print("Error: only one argument needed, the parameters json file.")

    if parameters_file == "--help" or len(sys.argv) < 2 or len(sys.argv) >= 5:
        print("################################################")
        print("### cd hypermapper")
        print("### Example 1: ")
        print(
            "### python3 hypermapper/compute_pareto.py example_scenarios/spatial/app_scenario.json"
        )
        print("### Example 2: ")
        print(
            "### python3 hypermapper/compute_pareto.py example_scenarios/spatial/app_scenario.json /path/to/input_data_file.csv /path/to/output_pareto_file.csv "
        )
        print("################################################")
        exit(1)

    compute(parameters_file, input_data_file, output_pareto_file)
