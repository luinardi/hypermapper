import warnings
from collections import OrderedDict

import matplotlib
import sys
import os

# ensure backward compatibility
try:
    from hypermapper.plot_hvi import plot_hvi
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

    from hypermapper.plot_hvi import plot_hvi
    from hypermapper.utility_functions import Logger

matplotlib.use("agg")


if __name__ == "__main__":
    try:
        from pygmo import hypervolume
    except ImportError as e:
        warnings.warn(
            "Failed to import pygmo. You can still use HyperMapper but plot_hvi.py won't work. To use it, please install pygmo according to https://esa.github.io/pygmo2/install.html .",
            ImportWarning,
            2,
        )
    warnings.warn(
        "Using 'scripts/plot_hvi' is deprecated and it will be removed in the future. Use 'hypermapper/plot_hvi' instead.",
        DeprecationWarning,
        2,
    )

    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the interactive mode we only want to log on the file.
    sys.stdout = Logger()

    list_of_dirs = []
    parameters_file = ""
    if len(sys.argv) >= 4:
        parameters_file = sys.argv[1]
        output_hvi_file_name = sys.argv[2]
        for dir in sys.argv[3:]:
            list_of_dirs.append(dir)
    else:
        print("Error: more arguments needed.")

    if parameters_file == "--help" or len(sys.argv) < 4:
        print("################################################")
        print("### Example: ")
        print("### cd hypermapper")
        print(
            "### python3 hypermapper/plot_hvi.py example_scenarios/spatial/DotProduct_scenario.json hvi_output_image_dotproduct.pdf /home/hypermapper_DotProduct /home/heuristic_DotProduct"
        )
        print("################################################")
        exit(1)

    plot_hvi(parameters_file, output_hvi_file_name, list_of_dirs)
    print("End of the plot_hvi script!")
