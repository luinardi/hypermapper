import argparse
import os
import sys
from os import listdir
from os.path import join

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

if not os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hypermapper.param import space
from hypermapper.util.logging import Logger
from hypermapper.util.file import read_settings_file, add_path


def load_data(file):
    data = pd.read_csv(file)

    data_array = {}
    for key in data:
        data_array[key] = data[key].tolist()
    return data_array


def plot_regret(
        settings_file,
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
    settings = read_settings_file(settings_file)
    run_directory = settings["run_directory"]
    if run_directory == ".":
        run_directory = os.environ["PWD"]
        settings["run_directory"] = run_directory
    log_file = settings["log_file"]
    log_file = add_path(settings, log_file)
    sys.stdout = Logger(log_file)
    param_space = space.Space(settings)
    output_metric = param_space.metric_names[0]  # only works for mono-objective
    doe_size = settings["design_of_experiment"]["number_of_samples"]
    feasibility_flag = param_space.feasible_output_name  # returns a list, we just want the name

    best = 0
    if minimum is not None:
        best = minimum
    application_name = settings["application_name"]

    if budget is None:
        budget = np.inf

    regrets = {}
    log_regrets = {}
    total_evaluations = {}
    max_iters = -np.inf
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
            incumbent = np.inf
            for idx in evaluations:
                if feasibility_flag is not None:
                    if data_array[feasibility_flag][idx] == True:
                        incumbent = min(incumbent, data_array[output_metric][idx])
                else:
                    incumbent = min(incumbent, data_array[output_metric][idx])
                regret = incumbent - best
                simple_regret.append((regret if regret != np.inf else np.nan))
            dir_regrets.append(np.array(simple_regret))

        for idx in range(len(dir_regrets)):
            dir_regrets[idx] = dir_regrets[idx][:min_dir_iters]

        regrets[data_dir] = np.array(dir_regrets)
        total_evaluations[data_dir] = list(range(min_dir_iters))

    mpl.rcParams.update({"font.size": 40})
    plt.rcParams["figure.figsize"] = [16, 12]
    linewidth = 2
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
        "violet",
        "lime",
    ]
    legend_elements = []
    if expert_configuration is not None:
        expert_data = [expert_configuration] * max_iters
        ax.plot(
            list(range(max_iters)),
            expert_data,
            color="black",
            linewidth=linewidth,
            linestyle="solid",
        )

    m = np.max([np.max(np.nanmax(regrets[key]), axis=0) for key in regrets.keys()])
    for key_idx, key in enumerate(regrets.keys()):
        regrets[key][np.isnan(regrets[key])] = m
        std = np.nanstd(regrets[key], axis=0, ddof=1)
        simple_means = np.nanmean(regrets[key], axis=0)
        lower_bound = []
        upper_bound = []
        plot_means = simple_means
        plot_stds = std

        for idx in range(plot_stds.shape[0]):
            lower_bound.append(plot_means[idx] - plot_stds[idx])
            upper_bound.append(plot_means[idx] + plot_stds[idx])

        next_color = colors[key_idx % len(colors)]
        x_values = np.array(total_evaluations[key]) + 1
        ax.plot(x_values, plot_means, color=next_color, linewidth=linewidth)
        ax.fill_between(
            x_values,
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
    if plot_log:
        ax.set_yscale("log")

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
        ax.axvline(x=doe_size, color="black", linewidth=linewidth, linestyle="dashed")

    rows = np.ceil(len(legend_elements) / ncol)
    height = 1 + (0.03) * rows
    ax.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, height),
        fancybox=True,
        shadow=True,
        ncol=ncol,
        bbox_transform=fig.transFigure,
    )

    if x_label is None:
        x_label = "Number of Evaluations"
    if y_label is None:
        if plot_log:
            y_label = "Log Regret"
        else:
            y_label = "Regret"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title is None:
        title = settings["application_name"]
    ax.set_title(title, y=1)

    ax.set_xlim(
        1,
    )

    if out_dir != "":
        if not out_dir.endswith("/"):
            out_dir += "/"
        os.makedirs(out_dir, exist_ok=True)
    if outfile is None:
        outfile = out_dir + application_name + "_regret.pdf"
    fig.savefig(outfile, bbox_inches="tight", dpi=300)
    fig.clear()

    return legend_elements


def main():
    parser = argparse.ArgumentParser(
        description="Plot regret for multiple optimization runs."
    )
    parser.add_argument(
        "--settings_file", "-j", dest="settings", action="store", default=None
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
    if args.settings is None or args.data_dirs is None:
        print("Error, needs settings file and at least one data dir")
        raise SystemExit

    plot_regret(
        args.settings,
        args.data_dirs,
        labels=args.labels,
        minimum=args.minimum,
        outfile=args.outfile,
        title=args.title,
        plot_log=args.plot_log,
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
