import argparse
import warnings
import sys
import os
from collections import OrderedDict

# ensure backward compatibility
try:
    from hypermapper.plot_optimization_results import plot_regret
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

    from hypermapper.plot_optimization_results import plot_regret

if __name__ == "__main__":
    warnings.warn(
        "Using 'scripts/plot_optimization_results' is deprecated and it will be removed in the future. "
        "Use 'hypermapper/plot_optimization_results' instead.",
        DeprecationWarning,
        2,
    )
    parser = argparse.ArgumentParser(
        description="Plot regret for multiple optimization runs."
    )
    parser.add_argument(
        "--configuration_file", "-j", dest="config", action="store", default=None
    )
    parser.add_argument(
        "--data_dirs", "-i", dest="data_dirs", action="store", default=None, nargs="+"
    )
    parser.add_argument(
        "--labels", "-l", dest="labels", action="store", default=None, nargs="+"
    )
    parser.add_argument(
        "--minimum", "-min", dest="minimum", action="store", default=0, type=int
    )
    parser.add_argument(
        "--output_dir", "-od", dest="out_dir", action="store", default=""
    )
    parser.add_argument(
        "--budget", "-b", dest="budget", action="store", type=int, default=None
    )
    parser.add_argument(
        "--output_file", "-o", dest="outfile", action="store", default=None
    )
    parser.add_argument("--title", "-t", dest="title", action="store", default=None)
    parser.add_argument("--plot_log", "-log", action="store_true")
    parser.add_argument("--unlog_y_axis", "-unlog", action="store_true")
    parser.add_argument("--show_doe", "-doe", action="store_true")
    parser.add_argument(
        "--x_label", "-xl", dest="x_label", action="store", default=None
    )
    parser.add_argument(
        "--y_label", "-yl", dest="y_label", action="store", default=None
    )
    parser.add_argument(
        "--ncol", "-nc", dest="ncol", action="store", type=int, default=4
    )
    parser.add_argument(
        "--expert_configuration",
        "-exp",
        dest="expert_configuration",
        action="store",
        type=float,
        default=None,
    )

    args = parser.parse_args()
    if args.config is None or args.data_dirs is None:
        print("Error, needs config file and at least one data dir")
        raise SystemExit

    plot_regret(
        args.config,
        args.data_dirs,
        labels=args.labels,
        minimum=args.minimum,
        outfile=args.outfile,
        title=args.title,
        plot_log=args.plot_log,
        unlog_y_axis=args.unlog_y_axis,
        budget=args.budget,
        out_dir=args.out_dir,
        ncol=args.ncol,
        x_label=args.x_label,
        y_label=args.y_label,
        show_doe=args.show_doe,
        expert_configuration=args.expert_configuration,
    )
