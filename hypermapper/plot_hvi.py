import json
from collections import OrderedDict, defaultdict

import matplotlib
from jsonschema import Draft4Validator
from pkg_resources import resource_stream

matplotlib.use("agg")  # noqa
import operator
import os
import sys
import warnings
import numpy as np

# ensure backward compatibility
try:
    from hypermapper.utility_functions import (
        validate_json,
        concatenate_data_dictionaries,
        compute_std_and_max_point,
        normalize_with_std,
        extend_with_default,
        Logger,
        deal_with_relative_and_absolute_path,
    )
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

    from hypermapper.utility_functions import (
        validate_json,
        concatenate_data_dictionaries,
        compute_std_and_max_point,
        normalize_with_std,
        extend_with_default,
        Logger,
        deal_with_relative_and_absolute_path,
    )

import matplotlib.pyplot as plt
from functools import reduce
from hypermapper import space
from os import listdir
from os.path import isfile, join
from copy import deepcopy

import scipy as sp
import scipy.stats

import csv
import math

debug = False


def compute_hvi(
    standard_deviation_optimization_metrics,
    input_files,
    dir,
    total_volume,
    max_point,
    hv_all_data,
    param_space,
    convert_in_seconds,
    number_of_bins,
):
    """
    Compute the HyperVolume Indicator (HVI) for a set of files specified in input_files contained in a directory dir.
    The HVI is computed as function of the timestamp present in the data.
    The data is divided in number_of_bins and the data the follow under a time bin is collected together.
    This will allow to plot blox plots.
    The invalid data is filtered.
    :param standard_deviation_optimization_metrics:
    :param input_files: the names of the csv files. The HVI is computed on each of these files.
    :param dir: the path of the directory where the csv files are stored.
    :param total_volume:
    :param max_point: this is potentially a multi-dimensional vector, one dimension for each optimization objectives.
    The max on all optimization dimensions for all the data concatenated considered in this plot is considered to compute
    this max_point. All the samples are smaller that this max_point.
    :param hv_all_data: this is the hypervolume of all the data concatenated considered in this plot.
    :param param_space: contains info on the input parameter space and the output parameter space.
    :param convert_in_seconds: if the timestamp is not in seconds this variable will convert it to seconds.
    Example: if timestamp is in milliseconds then you need to set convert_in_seconds = 1000.
    :param number_of_bins: number of bins to cluster the data. The final plot will be composed of this number of points in the x axis.
    :return: three objects: the first is the HVI computed for each bin and for each file: hvi[bin][file]; the second is the
    array that contains the temporal laps: bin_array_X[bin]. The third is the number_or_runs_in_bins which is a list of
    the number of DSE runs (or DSE repetitions) for a given dir that are actively sampling in that specific bin time interval.
    Sometime for example in a bin interval this number can be 0 because the active learning is taking a long time to compute
    and there are no new samples in that interval or because for some reason (system slowdown or else) at the end of
    the DSE some of the runs are faster so it is interesting to monitor how many are still running.
    """
    max_time = []
    reformatted_data_aux = {}
    reformatted_data_aux_timestamp = {}
    selection_keys_list = (
        param_space.get_output_parameters() + param_space.get_timestamp_parameter()
    )
    for file in input_files:
        full_path_file = dir + "/" + file
        feasible_flag = (
            True if (param_space.get_feasible_parameter() != [None]) else False
        )
        data_array, fast_addressing_of_data_array = param_space.load_data_file(
            full_path_file,
            selection_keys_list=selection_keys_list,
            only_valid=feasible_flag,
        )
        data_array[param_space.get_timestamp_parameter()[0]] = np.array(
            data_array[param_space.get_timestamp_parameter()[0]]
        )  # Transform list to np.array so that we can compute the max
        max_time.append(np.max(data_array[param_space.get_timestamp_parameter()[0]]))
        metric_data_aux = []
        for i, metric in enumerate(param_space.get_optimization_parameters()):
            X = np.array(data_array[metric])
            X /= standard_deviation_optimization_metrics[i]
            metric_data_aux.append(X)

        reformatted_data_aux_timestamp[file] = np.array(
            data_array[param_space.get_timestamp_parameter()[0]]
        )
        reformatted_data_aux[file] = list(zip(*metric_data_aux))

        if len(reformatted_data_aux[file]) == 0:
            print("Error: the data array is empty!")
            exit(1)

    # Compute global maximum time in seconds over the multiple files
    max_time_all = np.max(max_time)
    print("############# max_time_all %f" % max_time_all)

    # Decompose the data in number_of_bins bins
    bin_size = max_time_all / float(number_of_bins)
    bin_array_Y = {}
    bin_array_X = []
    tmp_min_bin = 0
    tmp_max_bin = tmp_min_bin + bin_size
    for bin in range(number_of_bins):
        bin_array_Y[bin] = {}
        for file in input_files:
            bin_array_Y[bin][file] = []

    for bin in range(number_of_bins):
        for file in input_files:
            for i, timestamp in enumerate(reformatted_data_aux_timestamp[file]):
                if timestamp >= tmp_min_bin and timestamp <= tmp_max_bin:
                    bin_array_Y[bin][file].append(tuple(reformatted_data_aux[file][i]))

        bin_array_X.append(tmp_max_bin)
        tmp_min_bin += bin_size
        tmp_max_bin = tmp_min_bin + bin_size

    for bin in range(number_of_bins):
        bin_array_X[bin] = bin_array_X[bin] / convert_in_seconds  # Convert in seconds

    comulative_bin = {}
    for file in input_files:
        comulative_bin[file] = []

    hvi = {}
    number_or_runs_in_bins = [
        0
    ] * number_of_bins  # Initialize list with number_of_bins of zeros.
    for file in input_files:
        hvi[file] = {}
        for bin in range(number_of_bins):
            comulative_bin[file] += bin_array_Y[bin][file]
            if bin_array_Y[bin][file] != []:
                number_or_runs_in_bins[bin] += 1
            if len(comulative_bin[file]) == 0:
                if bin == 0:
                    hvi[file][bin] = total_volume
                else:
                    hvi[file][bin] = hvi[file][bin - 1]
            else:
                hvi[file][bin] = HVI(comulative_bin[file], max_point, hv_all_data)

    return hvi, bin_array_X, number_or_runs_in_bins


# Hypervolume of objective space dominated by d
def H(d, r):
    try:
        from pygmo import hypervolume
    except ImportError as e:
        raise ImportError(
            "Failed to import pygmo. To use it, please install pygmo according to https://esa.github.io/pygmo2/install.html ."
        )
    return hypervolume(d).compute(r)


# Define hypervolume indicator function
def HVI(d, r, hv_all_data):
    hvi_tmp = hv_all_data - H(d, r)
    hvi_tmp = round(hvi_tmp, 14)  # Prevent issues from precision loss
    if hvi_tmp < 0:
        print("Error: HVI cannot be negative. Exit.")
        exit()
    return hvi_tmp


def HVI_from_files(real_pareto_file, parameters_file):
    """
    Compute hvi for a target Pareto front using the real Pareto front as reference.
    :param real_pareto_file: file containing the real Pareto front
    :param parameters_file: file containing the experiment scenario. Also used to find the target Pareto file.
    :return: the hvi of the target Pareto front
    """
    config = validate_json(parameters_file)
    param_space = space.Space(config)

    application_name = config["application_name"]
    test_pareto_file = config["output_pareto_file"]
    run_directory = config["run_directory"]
    if test_pareto_file == "output_pareto.csv":
        test_pareto_file = application_name + "_" + test_pareto_file
    test_pareto_file = deal_with_relative_and_absolute_path(
        run_directory, test_pareto_file
    )

    optimization_metrics = param_space.get_optimization_parameters()
    selection_keys = optimization_metrics + param_space.get_timestamp_parameter()
    feasible_flag = True if (param_space.get_feasible_parameter() != [None]) else False
    exhaustive_branin_pareto, _ = param_space.load_data_file(
        real_pareto_file, selection_keys_list=selection_keys, only_valid=feasible_flag
    )
    test_pareto, _ = param_space.load_data_file(
        test_pareto_file, selection_keys_list=selection_keys, only_valid=feasible_flag
    )
    concatenated_all_data_array = concatenate_data_dictionaries(
        exhaustive_branin_pareto, test_pareto, selection_keys_list=selection_keys
    )

    standard_deviations, max_point = compute_std_and_max_point(
        concatenated_all_data_array, optimization_metrics
    )

    exhaustive_branin_pareto = normalize_with_std(
        exhaustive_branin_pareto, standard_deviations, optimization_metrics
    )
    test_pareto = normalize_with_std(
        test_pareto, standard_deviations, optimization_metrics
    )

    exhaustive_branin_pareto = [
        exhaustive_branin_pareto[objective] for objective in optimization_metrics
    ]
    exhaustive_branin_pareto = list(zip(*exhaustive_branin_pareto))

    test_pareto = [test_pareto[objective] for objective in optimization_metrics]
    test_pareto = list(zip(*test_pareto))

    hv_exhaustive = H(exhaustive_branin_pareto, max_point)
    hv_test = H(test_pareto, max_point)
    hvi = hv_exhaustive - hv_test

    return hvi


# Get hypervolume indicator over each evaluation i of f(x), given initial data and evaluations data
def HVI_over_i(init, f_data, r, hv_all_data):
    d = deepcopy(init)
    y = [HVI(d, r, hv_all_data)]
    for j in range(len(f_data)):
        # d = np.row_stack([d, f_data[j]])
        d_list = d
        d_list.append(f_data[j])
        y.append(HVI(d_list, r, hv_all_data))
    return y


# Get x, y (HVI over i) for each array in the given list
def get_HVIs_for_datasets(ds, reformatted_all_data, r, hv_all_data):
    x, y = [], []
    for i in range(len(ds)):
        f_data = ds[i]
        x.append(list(range(len(f_data) + 1)))
        y.append(HVI_over_i(reformatted_all_data, f_data, r, hv_all_data))
    return x, y


def prod(ns):
    return reduce(operator.mul, ns, 1)


def lineplotCI(
    input_files,
    application_name,
    x_data,
    y_data,
    low_CI,
    upper_CI,
    xlabel,
    ylabel,
    title,
    output_filename="hvi_output_image.pdf",
):
    """
    Multiple line plots with intervals.

    :param input_files: list of directories. We want to plot one curve for each directory.
    :param application_name: name of the application. This is a string, ex: "DorProduct".
    :param x_data: a dictionary of directories. Each entry of the dictionary is an x array to plot.
    :param y_data: a dictionary of directories. Each entry of the dictionary is a y array to plot.
    :param low_CI: a dictionary of directories. Each entry of the dictionary is a lower confidence interval array to plot.
    :param upper_CI: a dictionary of directories. Each entry of the dictionary is a upper confidence interval array to plot.
    :param x_label: label x axis.
    :param y_label: label y axis.
    :param title: figure title. There is another title on the top of the figure that is given by the application name argument.
    :return:
    """

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    my_suptitle = fig.suptitle(application_name, fontsize=20, y=1.1)

    xlog = False
    ylog = True
    # Symlog sets a small interval near zero (both above and below) to use a linear scale.
    # This allows things to cross 0 without causing log(x) to explode (or go to -inf, rather).
    if xlog:
        ax.set_xscale("symlog")
        xlabel = "Log " + xlabel
    if ylog:
        ax.set_yscale("symlog")
        ylabel = "Log " + ylabel
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    for dir in input_files:
        name_line_legend = os.path.basename(dir)
        # Plot the data, set the linewidth, color and transparency of the line, provide a label for the legend
        ax.plot(x_data[dir], y_data[dir], lw=1, alpha=1, label=name_line_legend)
        # Shade the confidence interval
        ax.fill_between(x_data[dir], low_CI[dir], upper_CI[dir], alpha=0.4)

    ax.yaxis.grid(b=True, which="major", color="#CCCCCC", linestyle="-")
    ax.legend(loc="best")  # Display legend
    print("$$ Saving file " + output_filename)
    fig.savefig(
        output_filename, dpi=120, bbox_inches="tight", bbox_extra_artists=[my_suptitle]
    )


def boxplot(
    X,
    Y,
    application_name,
    number_of_bins,
    xlabel,
    ylabel,
    output_filename="boxplot.pdf",
):
    """
    :param xlabel:
    :param ylabel:
    :param X: the position on the X axis of the boxplots.
    :param Y: a list of lists where the list is the set of boxplots and the the lists are the values for each boxplot.
    :param application_name:
    :param number_of_bins:
    :param output_filename: the pdf file where the plot is saved
    :param output_filename:
    :return: save the boxplot in a pdf file, the name is specified in output_filename.
    """
    xlog = False
    ylog = True
    font = {"size": 16}

    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    # Symlog sets a small interval near zero (both above and below) to use a linear scale.
    # This allows things to cross 0 without causing log(x) to explode (or go to -inf, rather).
    if xlog:
        ax1.set_xscale("symlog")
        xlabel = "Log " + xlabel
    if ylog:
        ax1.set_yscale("symlog")
        ylabel = "Log " + ylabel
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.boxplot(Y)
    matplotlib.rc("font", **font)
    ax1.yaxis.grid(b=True, which="major", color="#CCCCCC", linestyle="-")
    i = 0
    final_array_ticks = []
    if len(X) != 0:
        for x in range(number_of_bins):
            if x in range(0, number_of_bins, int(number_of_bins / 20)):
                final_array_ticks.append(X[x])
                i += 1
            else:
                final_array_ticks.append("")

    ax1.set_xticklabels(final_array_ticks, rotation=45, fontsize=11)
    # Add the legend with some customizations.
    my_suptitle = fig.suptitle(
        "Five-number summary " + application_name, fontsize=20, y=1.1
    )
    print("$$ Saving file " + output_filename)
    fig.savefig(
        output_filename, dpi=120, bbox_inches="tight", bbox_extra_artists=[my_suptitle]
    )
    plt.clf()


def plot_hvi(parameters_file, output_hvi_file_name, list_of_dirs):
    """
    Plot the hypervolume indicator (HVI) results of the design space exploration.
    In this plot specifically we plot the HVI of HyperMapper's DSE against the HVI of a competing approach.
    On the x axis we plot time in seconds and on the y axis the HVI.
    HVI to be computed needs a real Pareto or at least a Pareto that is the best found by the results concatenation of
    HyperMapper and the competing approach.

    ######################################################
    ######### Input of this script ######################
    # 1) a file that is the real Pareto or the best Pareto found
    #    (supposing the we are comparing several approaches for example the best Pareto is the result of all these approaches combined).
    # 2) a file containing all the samples of the exploration (not only the Pareto).
    #    From this file we can compute the Pareto at time t and then the hvi at time t
    """
    try:
        import statsmodels.stats.api as sms
    except:
        # TODO: Long-term: move this import to the top.
        ImportError(
            "Failed to import statsmodels. Statsmodels is required for plot_hvi."
        )
    xlabel = "Time (sec)"
    ylabel = "HyperVolume Indicator (HVI)"
    number_of_bins = 20

    filename, file_extension = os.path.splitext(parameters_file)
    if file_extension != ".json":
        print(
            "Error: invalid file name. \nThe input file has to be a .json file not a %s"
            % file_extension
        )
        exit(1)
    with open(parameters_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("hypermapper", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    DefaultValidatingDraft4Validator(schema).validate(config)

    if "application_name" in config:
        application_name = config["application_name"]
    else:
        application_name = ""

    print("########## plot_hvi.py #########################")
    print("### Parameters file is %s" % parameters_file)
    print("### Application name is %s" % application_name)
    print("### The input directories data are %s" % str(list_of_dirs))
    print("################################################")

    param_space = space.Space(config)
    optimization_metrics = param_space.get_optimization_parameters()

    ###################################################################################################################
    ########### Compute the hypervolume of all the input files concatenated as a reference for the HVI metric.
    ###################################################################################################################
    input_files = {}

    # y_data_mean is dict on the directories that for each entry in the dict contains the mean of each point x over multiple file repetitions in one directory; lower and upper are for the confidence interval.
    y_data_mean = defaultdict(list)
    y_data_median = defaultdict(list)
    y_data_min = defaultdict(list)
    y_data_max = defaultdict(list)
    y_data_lower = defaultdict(list)
    y_data_upper = defaultdict(list)
    bin_array_X = {}
    number_or_runs_in_bins = {}

    for dir in list_of_dirs:
        input_files[dir] = [f for f in listdir(dir) if isfile(join(dir, f))]

    for dir in list_of_dirs:
        files_to_remove = []
        for file in input_files[dir]:
            filename, file_extension = os.path.splitext(file)
            if file_extension != ".csv":
                print(
                    "Warning: file %s is not a csv file, it will not be considered in the HVI plot. "
                    % file
                )
                files_to_remove.append(file)
        # Don't move this for loop inside the previous identical one otherwise you will remove the elements before they get process because of overlapping references.
        for file in files_to_remove:
            input_files[dir].remove(file)

    for dir in list_of_dirs:
        if len(input_files[dir]) == 0:
            print(
                "Warning: directory %s is empty, it will not be considered in the HVI plot."
            )
            del input_files[dir]

    if len(input_files) == 0:
        print("Error: there no input files to compute the HVI.")

    print("The files used as a input are: ")
    for i, dir in enumerate(input_files.keys()):
        print(
            "Directory "
            + str(i)
            + ": "
            + dir
            + ", # of files: "
            + str(len(input_files[dir]))
            + ", list of files: "
            + str(input_files[dir])
        )

    all_data_files = []
    for dir in input_files.keys():
        for file in input_files[dir]:
            all_data_files += [dir + "/" + file]

    selection_keys = (
        param_space.get_output_parameters() + param_space.get_timestamp_parameter()
    )
    feasible_flag = True if (param_space.get_feasible_parameter() != [None]) else False
    concatenated_all_data_array = param_space.load_data_files(
        all_data_files, selection_keys_list=selection_keys, only_valid=feasible_flag
    )

    if len(next(iter(concatenated_all_data_array.values()))) == 0:
        return return_empty_images(
            application_name,
            input_files,
            number_of_bins,
            output_hvi_file_name,
            xlabel,
            ylabel,
        )

    bounds = {}
    max_point = []
    standard_deviation_optimization_metrics = []
    max_min_difference = []
    # Get bounds of objective space
    for metric in optimization_metrics:
        X = np.array(concatenated_all_data_array[metric])

        standard_deviation = np.std(X, axis=0)
        standard_deviation_optimization_metrics.append(standard_deviation)
        X /= standard_deviation

        concatenated_all_data_array[metric] = X
        bounds[metric] = (
            min(concatenated_all_data_array[metric]),
            max(concatenated_all_data_array[metric]),
        )
        max_point.append(bounds[metric][1])
        max_min_difference.append(bounds[metric][1] - bounds[metric][0])
        print(
            "(min, max) = (%f, %f) for the metric %s. This is to compute the hypervolume."
            % (bounds[metric][0], bounds[metric][1], metric)
        )

    total_volume = prod(max_min_difference)
    list_of_objectives = [
        concatenated_all_data_array[objective]
        for objective in param_space.get_optimization_parameters()
    ]
    reformatted_all_data = list(zip(*list_of_objectives))

    # Get dominated hypervolume for Pareto of all data observed
    hv_all_data = H(reformatted_all_data, max_point)
    print("The hypervolume of all the files concatenated: %d" % hv_all_data)

    ###################################################################################################################
    ########### Compute the HVI for each directory.
    ###################################################################################################################
    hvi = {}
    for dir in input_files:
        print("Compute HVI for %s" % dir)
        convert_in_seconds = 1000.0
        hvi[dir], bin_array_X[dir], number_or_runs_in_bins[dir] = compute_hvi(
            standard_deviation_optimization_metrics,
            input_files[dir],
            dir,
            total_volume,
            max_point,
            hv_all_data,
            param_space,
            convert_in_seconds,
            number_of_bins,
        )

        # Round the floating point numbers to 1 decimal for clarity of visualization.
        bin_array_X[dir] = [round(float(i), 1) for i in bin_array_X[dir]]
        for file in hvi[dir]:
            for bin in hvi[dir][file]:
                hvi[dir][file][bin] = round(float(hvi[dir][file][bin]), 1)

    ###################################################################################################################
    ########### Plot all the HVIs (using box plots bin_array_X and hvi)
    ###################################################################################################################

    for dir in input_files:
        hvi_list_of_lists = []
        each_bin = defaultdict(list)
        for file in hvi[dir]:
            for bin in hvi[dir][file]:
                each_bin[bin].append(hvi[dir][file][bin])
        for bin in hvi[dir][file]:
            hvi_list_of_lists.append(
                each_bin[bin]
            )  # This is a list of bins and for each bin there is a list of hvi values for each file in that directory.

        # Print boxplot (one figure per directory).
        boxplot(
            bin_array_X[dir],
            hvi_list_of_lists,
            application_name,
            number_of_bins,
            xlabel,
            ylabel,
            str(dir + "/" + os.path.basename(dir) + "_boxplot" + ".pdf"),
        )

        # Print lineplot (only one figure comparing all the directories).
        for hvi_list in hvi_list_of_lists:
            hvi_list_array = np.array(hvi_list)
            y_data_mean[dir].append(hvi_list_array.mean())
            y_data_median[dir].append(np.median(hvi_list_array))
            y_data_min[dir].append(np.min(hvi_list_array))
            y_data_max[dir].append(np.max(hvi_list_array))
            low, up = sms.DescrStatsW(hvi_list_array).tconfint_mean()
            y_data_lower[dir].append(low)
            y_data_upper[dir].append(up)

        for bin_number, bin_value in enumerate(y_data_lower[dir]):
            if not math.isnan(bin_value) and bin_value < 0:
                y_data_lower[dir][bin_number] = 0
        for bin_number, bin_value in enumerate(y_data_upper[dir]):
            if not math.isnan(bin_value) and bin_value < 0:
                y_data_upper[dir][bin_number] = 0

        print_stats_on_a_txt(
            dir,
            str(dir + "/" + os.path.basename(dir) + "_stats" + ".txt"),
            bin_array_X,
            number_or_runs_in_bins,
            y_data_mean,
            y_data_median,
            y_data_min,
            y_data_max,
            y_data_lower,
            y_data_upper,
        )

    # Call the function to create plot
    lineplotCI(
        input_files,
        application_name,
        x_data=bin_array_X,
        y_data=y_data_mean,
        low_CI=y_data_lower,
        upper_CI=y_data_upper,
        xlabel=xlabel,
        ylabel=ylabel,
        title="Line plot with 95% confidence intervals",
        output_filename=output_hvi_file_name,
    )


def return_empty_images(
    application_name, input_files, number_of_bins, output_hvi_file_name, xlabel, ylabel
):
    """
    This function deals with some extreme case where the files are empty.
    :param application_name:
    :param input_files:
    :param number_of_bins:
    :param output_hvi_file_name:
    :param xlabel:
    :param ylabel:
    :return: empty images and stat files.
    """

    y_data_mean = defaultdict(list)
    y_data_median = defaultdict(list)
    y_data_min = defaultdict(list)
    y_data_max = defaultdict(list)
    y_data_lower = defaultdict(list)
    y_data_upper = defaultdict(list)
    bin_array_X = {}
    number_or_runs_in_bins = {}

    print(
        "Warning: the hypervolume of all the files concatenated is undefined, the files are empty."
    )
    # Print empty boxplot (one figure per directory).
    for dir in input_files:
        bin_array_X[dir] = []
        number_or_runs_in_bins[dir] = []
        y_data_mean[dir] = []
        y_data_median[dir] = []
        y_data_min[dir] = []
        y_data_max[dir] = []
        y_data_lower[dir] = []
        y_data_upper[dir] = []
        boxplot(
            bin_array_X[dir],
            [[]],
            application_name,
            number_of_bins,
            xlabel,
            ylabel,
            str(dir + "/" + os.path.basename(dir) + "_boxplot" + ".pdf"),
        )

        print_stats_on_a_txt(
            dir,
            str(dir + "/" + os.path.basename(dir) + "_stats" + ".txt"),
            bin_array_X,
            number_or_runs_in_bins,
            y_data_mean,
            y_data_median,
            y_data_min,
            y_data_max,
            y_data_lower,
            y_data_upper,
        )
    # Call the function to create plot
    lineplotCI(
        input_files,
        application_name,
        x_data=bin_array_X,
        y_data=y_data_mean,
        low_CI=y_data_lower,
        upper_CI=y_data_upper,
        xlabel=xlabel,
        ylabel=ylabel,
        title="Line plot with 95% confidence intervals",
        output_filename=output_hvi_file_name,
    )


def print_stats_on_a_txt(
    dir,
    filename,
    bin_array_X,
    number_or_runs_in_bins,
    y_data_mean,
    y_data_median,
    y_data_min,
    y_data_max,
    y_data_lower,
    y_data_upper,
):
    """
    Print stats on a txt file under the form of a csv file. The stats here are roughly the same as the boxplot but in a
    text form with some additional field.
    :param dir: the directory's files we want the stats of.
    :param filename: the name of the txt file where to save the results.
    :param bin_array_X: this is a dictionary of dirs and bins which represents the x axis value of the HVI,
    this is the time where the DSE achieved a level of HVI.
    :param number_or_runs_in_bins: this is a list of the number of DSE runs (or DSE repetitions) for a given dir that
    are actively sampling in that specific bin time interval. Sometime for example in a bin interval this number can be
    0 because the active learning is taking a long time to compute and there are no new samples in that interval or
    because for some reason (system slowdown or else) at the end of the DSE some of the runs are faster so it is interesting
    to monitor how many are still running.
    :param y_data_mean: this is a dictionary of dirs and bins which represents the mean of the HVI.
    :param y_data_median: this is a dictionary of dirs and bins which represents the median of the HVI.
    :param y_data_min: this is a dictionary of dirs and bins which represents the min of the HVI.
    :param y_data_max: this is a dictionary of dirs and bins which represents the max of the HVI.
    :param y_data_lower: this is a dictionary of dirs and bins which represents the lower confidence interval (CI) of the HVI.
    :param y_data_upper: this is a dictionary of dirs and bins which represents the upper confidence interval (CI) of the HVI.
    :return: save the file in filename on disk.
    """
    with open(filename, "w") as f:
        w = csv.writer(f)
        headers = [
            "Bucket #",
            "Time sec",
            "# of runs in bin",
            "HVI mean",
            "HVI median",
            "HVI min",
            "HVI max",
            "HVI 95% CI lower bound",
            "HVI 95% CI upper bound",
        ]
        w.writerow(headers)
        for bin in range(len(bin_array_X[dir])):
            row = []
            row.append(str(bin + 1))
            row.append(str(bin_array_X[dir][bin]))
            row.append(str(number_or_runs_in_bins[dir][bin]))
            row.append(str(y_data_mean[dir][bin]))
            row.append(str(y_data_median[dir][bin]))
            row.append(str(y_data_min[dir][bin]))
            row.append(str(y_data_max[dir][bin]))
            row.append(str(y_data_lower[dir][bin]))
            row.append(str(y_data_upper[dir][bin]))
            w.writerow(row)
    print("$$ Saving file " + filename)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def main():
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
        print(
            "### hm-plot-hvi example_scenarios/spatial/DotProduct_scenario.json hvi_output_image_dotproduct.pdf /home/hypermapper_DotProduct /home/heuristic_DotProduct"
        )
        print("################################################")
        exit(1)

    plot_hvi(parameters_file, output_hvi_file_name, list_of_dirs)
    print("End of the plot_hvi script!")


if __name__ == "__main__":
    main()
