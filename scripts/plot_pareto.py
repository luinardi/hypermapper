import json
import warnings
import sys
import os
from collections import OrderedDict, defaultdict

import matplotlib
from jsonschema import Draft4Validator
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

matplotlib.use("agg")  # noqa

# ensure backward compatibility
try:
    from hypermapper import space
    from hypermapper.plot_pareto import plot, debug
    from hypermapper.utility_functions import (
        Logger,
        deal_with_relative_and_absolute_path,
        extend_with_default,
        get_next_color,
        get_last_dir_and_file_names,
    )
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

    from hypermapper import space
    from hypermapper.plot_pareto import plot, debug
    from hypermapper.utility_functions import (
        Logger,
        deal_with_relative_and_absolute_path,
        extend_with_default,
        get_next_color,
        get_last_dir_and_file_names,
    )

if __name__ == "__main__":
    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the interactive mode we only want to log on the file.
    sys.stdout = Logger()

    list_of_pairs_of_files = []
    image_output_file = None
    parameters_file = ""
    if len(sys.argv) >= 2:
        parameters_file = sys.argv[1]
        if len(sys.argv) >= 3:
            i = 2
            try:
                image_output_file = sys.argv[i]
                filename, file_extension = os.path.splitext(
                    image_output_file
                )  # Test on the file to have a pdf extension
                if file_extension != ".pdf":
                    print(
                        "Error: file extension has to be a pdf. Given: %s"
                        % file_extension
                    )
                    exit(1)
                i += 1
            except:
                print("Error reading the image name file for arguments.")
                exit(1)

            while i < len(sys.argv):
                try:
                    list_of_pairs_of_files.append((sys.argv[i], sys.argv[i + 1]))
                except:
                    print(
                        "Error: wrong number of files. Files have to be in pairs of pareto and search."
                    )
                    exit(1)
                i += 2
    else:
        print("Error: more arguments needed.")

    if parameters_file == "--help" or len(sys.argv) < 2:
        print("################################################")
        print("### cd hypemapper")
        print("### Example 1: ")
        print(
            "### python3 hypermapper/plot_pareto.py example_scenarios/spatial/BlackScholes_scenario.json"
        )
        print("### Example 2: ")
        print(
            "### python3 hypermapper/plot_pareto.py example_scenarios/spatial/BlackScholes_scenario.json /path/to/output/image.pdf file1_pareto file1_search file2_pareto file2_search file3_pareto file3_search"
        )
        print("################################################")
        exit(1)

    plot(parameters_file, list_of_pairs_of_files, image_output_file)
    print("End of the plot_pareto script!")
