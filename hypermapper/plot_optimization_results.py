import argparse
import json
import os
import sys
import warnings
import numpy as np
from collections import OrderedDict
from os import listdir
from os.path import join, splitext

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from jsonschema import exceptions, Draft4Validator
from matplotlib.lines import Line2D
from pkg_resources import resource_stream

# ensure backward compatibility
try:
    from hypermapper import space
    from hypermapper.utility_functions import extend_with_default
except ImportError:
    if os.getenv("HYPERMAPPER_HOME"):  # noqa
        warnings.warn(
            "Found environment variable 'HYPERMAPPER_HOME', used to update the system path. Support might be discontinued in the future. Please make sure your installation is working without this environment variable, e.g., by installing with 'pip install hypermapper'.",
            DeprecationWarning,
            2,
        )  # noqa
        sys.path.append(os.environ["HYPERMAPPER_HOME"])  # noqa
    ppath = os.getenv("PYTHONPATH")
    if ppath:
        path_items = ppath.split(":")

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

    sys.path.append(".")  # noqa
    sys.path = list(OrderedDict.fromkeys(sys.path))

    from hypermapper import space
    from hypermapper.utility_functions import extend_with_default


def load_data(file):
    data = pd.read_csv(file)

    data_array = {}
    for key in data:
        data_array[key] = data[key].tolist()
    return data_array


def plot_regret(
    configuration_file,
    data_dirs,
    labels=None,
    minimum=0,
    outfile=None,
    title=None,
    plot_log=False,
    unlog_y_axis=False,
    budget=None,
    out_dir=None,
    ncol=4,
    x_label=None,
    y_label=None,
    show_doe=True,
    expert_configuration=None,
):
    # Read json configuration file
    if not configuration_file.endswith(".json"):
        _, file_extension = splitext(configuration_file)
        print(
            "Error: invalid file name. \nThe input file has to be a .json file not a %s"
            % file_extension
        )
        raise SystemExit
    with open(configuration_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    try:
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    param_space = space.Space(config)
    output_metric = param_space.get_optimization_parameters()[
        0
    ]  # only works for mono-objective
    doe_size = config["design_of_experiment"]["number_of_samples"]
    feasibility_flag = param_space.get_feasible_parameter()[
        0
    ]  # returns a list, we just want the name

    best = 0
    if minimum is not None:
        best = minimum
    application_name = config["application_name"]

    if budget is None:
        budget = float("inf")

    regrets = {}
    log_regrets = {}
    total_evaluations = {}
    max_iters = float("-inf")
    for data_dir_idx, data_dir in enumerate(data_dirs):
        dir_regrets = []
        dir_log_regrets = []
        min_dir_iters = budget
        for file in listdir(data_dir):
            if not file.endswith(".csv"):
                print("Skipping:", file)
                continue
            full_file = join(data_dir, file)
            data_array = load_data(full_file)
            total_iters = min(len(data_array[output_metric]), budget)
            min_dir_iters = min(total_iters, min_dir_iters)
            max_iters = max(max_iters, total_iters)
            evaluations = list(range(total_iters))
            simple_regret = []
            log_regret = []
            incumbent = float("inf")
            for idx in evaluations:
                if feasibility_flag is not None:
                    if data_array[feasibility_flag][idx] == True:
                        incumbent = min(incumbent, data_array[output_metric][idx])
                else:
                    incumbent = min(incumbent, data_array[output_metric][idx])
                regret = incumbent - best
                simple_regret.append(regret)
                log_regret.append(np.log(regret))
            dir_regrets.append(np.array(simple_regret))
            dir_log_regrets.append(np.array(log_regret))

        for idx in range(len(dir_regrets)):
            dir_regrets[idx] = dir_regrets[idx][:min_dir_iters]
            dir_log_regrets[idx] = dir_log_regrets[idx][:min_dir_iters]

        regrets[data_dir] = np.array(dir_regrets)
        log_regrets[data_dir] = np.array(dir_log_regrets)
        total_evaluations[data_dir] = list(range(min_dir_iters))

    mpl.rcParams.update({"font.size": 40})
    plt.rcParams["figure.figsize"] = [16, 12]
    linewidth = 6
    fig, ax = plt.subplots()
    colors = [
        "blue",
        "green",
        "red",
        "magenta",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "gray",
    ]
    legend_elements = []
    if expert_configuration is not None:
        if plot_log:
            expert_configuration = np.log(expert_configuration)
        expert_data = [expert_configuration] * max_iters
        plt.plot(
            list(range(max_iters)),
            expert_data,
            color="black",
            linewidth=linewidth,
            linestyle="solid",
        )

    for key_idx, key in enumerate(regrets.keys()):
        std = np.std(regrets[key], axis=0, ddof=1)
        log_std = np.std(log_regrets[key], axis=0, ddof=1)
        simple_means = np.mean(regrets[key], axis=0)
        log_means = np.log(simple_means)
        lower_bound = []
        upper_bound = []
        plot_means = simple_means
        plot_stds = std
        if plot_log:
            plot_means = log_means
            plot_stds = log_std

        for idx in range(plot_stds.shape[0]):
            lower_bound.append(plot_means[idx] - plot_stds[idx])
            upper_bound.append(plot_means[idx] + plot_stds[idx])

        next_color = colors[key_idx % len(colors)]
        plt.plot(
            total_evaluations[key], plot_means, color=next_color, linewidth=linewidth
        )
        plt.fill_between(
            total_evaluations[key],
            lower_bound,
            upper_bound,
            color=next_color,
            alpha=0.2,
        )

        if labels is None:
            label = key
        else:
            label = labels[key_idx]

        legend_elements.append(
            Line2D([0], [0], color=next_color, label=label, linewidth=linewidth)
        )

    if expert_configuration is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=linewidth,
                linestyle="solid",
                label="Expert Configuration",
            )
        )

    if plot_log and unlog_y_axis:
        locs, plt_labels = plt.yticks()
        plt_labels = [np.exp(float(item)) for item in locs]
        plt_labels = ["{0:,.2f}\n".format(item) for item in plt_labels]
        plt.yticks(locs, plt_labels)

    if show_doe:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=linewidth,
                linestyle="dashed",
                label="Initialization",
            )
        )
        plt.axvline(x=doe_size, color="black", linewidth=linewidth, linestyle="dashed")

    rows = np.ceil(len(legend_elements) / ncol)
    height = 1 + (0.03) * rows
    plt.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, height),
        fancybox=True,
        shadow=True,
        ncol=ncol,
        bbox_transform=plt.gcf().transFigure,
    )

    if x_label is None:
        x_label = "Number of Evaluations"
    if y_label is None:
        if plot_log:
            y_label = "Log Regret"
        else:
            y_label = "Regret"
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title is None:
        title = config["application_name"]
    plt.title(title, y=1)

    plt.xlim(
        1,
    )

    if out_dir != "":
        if not out_dir.endswith("/"):
            out_dir += "/"
        os.makedirs(out_dir, exist_ok=True)
    if outfile is None:
        outfile = out_dir + application_name + "_regret.pdf"
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.gcf().clear()

    return legend_elements


def main():
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


if __name__ == "__main__":
    main()
