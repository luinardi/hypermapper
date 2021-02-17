import json
import warnings
import sys
import os
from collections import OrderedDict

from jsonschema import exceptions, Draft4Validator
from pkg_resources import resource_stream

# ensure backward compatibility
try:
    from hypermapper.evolution import main
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        Logger,
        extend_with_default,
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

    from hypermapper.evolution import main
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        Logger,
        extend_with_default,
    )

if __name__ == "__main__":
    warnings.warn(
        "Using 'scripts/evolution' is deprecated and it will be removed in the future. Use 'hypermapper/evolution' instead.",
        DeprecationWarning,
        2,
    )

    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the client-server mode we only want to log on the file.
    # This is in local search but not random_scalarizations. I leave it out for now
    # sys.stdout = Logger()

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
        raise SystemExit

    try:
        initial_directory = os.environ["PWD"]
        hypermapper_home = os.environ["HYPERMAPPER_HOME"]
        os.chdir(hypermapper_home)
    except:
        hypermapper_home = "."
        initial_directory = "."

    if not parameters_file.endswith(".json"):
        _, file_extension = os.path.splitext(parameters_file)
        print(
            "Error: invalid file name. \nThe input file has to be a .json file not a %s"
            % file_extension
        )
        raise SystemExit
    with open(parameters_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

    try:
        DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    run_directory = config["run_directory"]
    if run_directory == ".":
        run_directory = initial_directory
        config["run_directory"] = run_directory
    log_file = config["log_file"]
    if log_file == "hypermapper_logfile.log":
        log_file = deal_with_relative_and_absolute_path(run_directory, log_file)
    sys.stdout = Logger(log_file)

    main(config)

    try:
        os.chdir(hypermapper_pwd)
    except:
        pass

    print("### End of the evolutionary script.")
