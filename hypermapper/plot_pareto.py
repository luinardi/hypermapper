"""
Plots design space exploration results.
"""
import json
from collections import OrderedDict, defaultdict

import matplotlib
from jsonschema import Draft4Validator
from pkg_resources import resource_stream

matplotlib.use("agg")  # noqa
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import os
import sys
import warnings

# ensure backward compatibility
try:
    from hypermapper import space
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        get_next_color,
        get_last_dir_and_file_names,
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
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        get_next_color,
        get_last_dir_and_file_names,
        Logger,
        extend_with_default,
    )

debug = False


def plot(parameters_file, list_of_pairs_of_files=[], image_output_file=None):
    """
    Plot the results of the previously run design space exploration.
    """
    try:
        hypermapper_pwd = os.environ["PWD"]
        hypermapper_home = os.environ["HYPERMAPPER_HOME"]
        os.chdir(hypermapper_home)
    except:
        hypermapper_home = "."
        hypermapper_pwd = "."
    show_samples = False

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

    application_name = config["application_name"]
    optimization_metrics = config["optimization_objectives"]
    feasible_output = config["feasible_output"]
    feasible_output_name = feasible_output["name"]
    run_directory = config["run_directory"]
    if run_directory == ".":
        run_directory = hypermapper_pwd
        config["run_directory"] = run_directory

    xlog = config["output_image"]["image_xlog"]
    ylog = config["output_image"]["image_ylog"]

    if "optimization_objectives_labels_image_pdf" in config["output_image"]:
        optimization_objectives_labels_image_pdf = config["output_image"][
            "optimization_objectives_labels_image_pdf"
        ]
    else:
        optimization_objectives_labels_image_pdf = optimization_metrics

    # Only consider the files in the json file if there are no input files.
    if list_of_pairs_of_files == []:
        output_pareto_file = config["output_pareto_file"]
        if output_pareto_file == "output_pareto.csv":
            output_pareto_file = application_name + "_" + output_pareto_file
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
        list_of_pairs_of_files.append(
            (
                deal_with_relative_and_absolute_path(run_directory, output_pareto_file),
                deal_with_relative_and_absolute_path(run_directory, output_data_file),
            )
        )
    else:
        for idx, (output_pareto_file, output_data_file) in enumerate(
            list_of_pairs_of_files
        ):
            list_of_pairs_of_files[idx] = (
                deal_with_relative_and_absolute_path(run_directory, output_pareto_file),
                deal_with_relative_and_absolute_path(run_directory, output_data_file),
            )

    if image_output_file != None:
        output_image_pdf_file = image_output_file
        output_image_pdf_file = deal_with_relative_and_absolute_path(
            run_directory, output_image_pdf_file
        )
        filename = os.path.basename(output_image_pdf_file)
        path = os.path.dirname(output_image_pdf_file)
        if path == "":
            output_image_pdf_file_with_all_samples = "all_" + filename
        else:
            output_image_pdf_file_with_all_samples = path + "/" + "all_" + filename
    else:
        tmp_file_name = config["output_image"]["output_image_pdf_file"]
        if tmp_file_name == "output_pareto.pdf":
            tmp_file_name = application_name + "_" + tmp_file_name
        output_image_pdf_file = deal_with_relative_and_absolute_path(
            run_directory, tmp_file_name
        )
        filename = os.path.basename(output_image_pdf_file)
        path = os.path.dirname(output_image_pdf_file)
        if path == "":
            output_image_pdf_file_with_all_samples = "all_" + filename
        else:
            output_image_pdf_file_with_all_samples = path + "/" + "all_" + filename

    str_files = ""
    for e in list_of_pairs_of_files:
        str_files += str(e[0] + " " + e[1] + " ")

    print("######### plot_pareto.py ##########################")
    print("### Parameters file is %s" % parameters_file)
    print("### The Pareto and DSE data files are: %s" % str_files)
    print("### The first output pdf image is %s" % output_image_pdf_file)
    print(
        "### The second output pdf image is %s" % output_image_pdf_file_with_all_samples
    )
    print("################################################")

    param_space = space.Space(config)

    xelem = optimization_metrics[0]
    yelem = optimization_metrics[1]
    handler_map_for_legend = {}
    xlabel = optimization_objectives_labels_image_pdf[0]
    ylabel = optimization_objectives_labels_image_pdf[1]

    x_max = float("-inf")
    x_min = float("inf")
    y_max = float("-inf")
    y_min = float("inf")

    print_legend = True
    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    if xlog:
        ax1.set_xscale("log")
    if ylog:
        ax1.set_yscale("log")

    objective_1_max = objective_2_max = 1
    objective_1_is_percentage = objective_2_is_percentage = False
    if "objective_1_max" in config["output_image"]:
        objective_1_max = config["output_image"]["objective_1_max"]
        objective_1_is_percentage = True
    if "objective_2_max" in config["output_image"]:
        objective_2_max = config["output_image"]["objective_2_max"]
        objective_2_is_percentage = True

    input_data_array = {}
    fast_addressing_of_data_array = {}
    non_valid_optimization_obj_1 = defaultdict(list)
    non_valid_optimization_obj_2 = defaultdict(list)

    for (
        file_pair
    ) in (
        list_of_pairs_of_files
    ):  # file_pair is tuple containing: (pareto file, DSE file)
        next_color = get_next_color()

        #############################################################################
        ###### Load data from files and do preprocessing on the data before plotting.
        #############################################################################
        for file in file_pair:
            print(("Loading data from %s ..." % file))
            (
                input_data_array[file],
                fast_addressing_of_data_array[file],
            ) = param_space.load_data_file(file, debug)
            if input_data_array[file] == None:
                print("Error: no data found in input data file: %s. \n" % file_pair[1])
                exit(1)
            if (xelem not in input_data_array[file]) or (
                yelem not in input_data_array[file]
            ):
                print(
                    "Error: the optimization variables have not been found in input data file %s. \n"
                    % file
                )
                exit(1)
            print(("Parameters are " + str(list(input_data_array[file].keys())) + "\n"))
            input_data_array[file][xelem] = [
                float(input_data_array[file][xelem][i]) / objective_1_max
                for i in range(len(input_data_array[file][xelem]))
            ]
            input_data_array[file][yelem] = [
                float(input_data_array[file][yelem][i]) / objective_2_max
                for i in range(len(input_data_array[file][yelem]))
            ]

            if objective_1_is_percentage:
                input_data_array[file][xelem] = [
                    input_data_array[file][xelem][i] * 100
                    for i in range(len(input_data_array[file][xelem]))
                ]
            if objective_2_is_percentage:
                input_data_array[file][yelem] = [
                    input_data_array[file][yelem][i] * 100
                    for i in range(len(input_data_array[file][yelem]))
                ]

            x_max, x_min, y_max, y_min = compute_min_max_samples(
                input_data_array[file], x_max, x_min, xelem, y_max, y_min, yelem
            )

            input_data_array_size = len(
                input_data_array[file][list(input_data_array[file].keys())[0]]
            )
            print("Size of the data file %s is %d" % (file, input_data_array_size))

        file_pareto = file_pair[0]  # This is the Pareto file
        file_search = file_pair[1]  # This is the DSE file

        ######################################################################################################
        ###### Compute invalid samples to be plot in a different color (and remove them from the data arrays).
        ######################################################################################################
        if show_samples:
            i = 0
            for ind in range(len(input_data_array[file][yelem])):
                if input_data_array[file][feasible_output_name][i] == False:
                    non_valid_optimization_obj_2[file_search].append(
                        input_data_array[file][yelem][i]
                    )
                    non_valid_optimization_obj_1[file_search].append(
                        input_data_array[file][xelem][i]
                    )
                    for key in list(input_data_array[file].keys()):
                        del input_data_array[file][key][i]
                else:
                    i += 1

            label_is = get_last_dir_and_file_names(file_pareto)
            (all_samples,) = plt.plot(
                input_data_array[file_search][xelem],
                input_data_array[file_search][yelem],
                color=next_color,
                linestyle="None",
                marker=".",
                mew=0.5,
                markersize=3,
                fillstyle="none",
                label=label_is,
            )
            plt.plot(
                input_data_array[file_pareto][xelem],
                input_data_array[file_pareto][yelem],
                linestyle="None",
                marker=".",
                mew=0.5,
                markersize=3,
                fillstyle="none",
            )
            handler_map_for_legend[all_samples] = HandlerLine2D(numpoints=1)

        ################################################################################################################
        ##### Create a straight Pareto plot: we need to add one point for each point of the data in paretoX and paretoY.
        ##### We also need to reorder the points on the x axis first.
        ################################################################################################################
        straight_pareto_x = list()
        straight_pareto_y = list()
        if len(input_data_array[file_pareto][xelem]) != 0:
            data_array_pareto_x, data_array_pareto_y = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(
                            input_data_array[file_pareto][xelem],
                            input_data_array[file_pareto][yelem],
                        )
                    )
                )
            )
            for j in range(len(data_array_pareto_x)):
                straight_pareto_x.append(data_array_pareto_x[j])
                straight_pareto_x.append(data_array_pareto_x[j])
                straight_pareto_y.append(data_array_pareto_y[j])
                straight_pareto_y.append(data_array_pareto_y[j])
            straight_pareto_x.append(x_max)  # Just insert the max on the x axis
            straight_pareto_y.insert(0, y_max)  # Just insert the max on the y axis

        label_is = "Pareto - " + get_last_dir_and_file_names(file_pareto)

        (pareto_front,) = plt.plot(
            straight_pareto_x,
            straight_pareto_y,
            label=label_is,
            linewidth=1,
            color=next_color,
        )
        handler_map_for_legend[pareto_front] = HandlerLine2D(numpoints=1)

        label_is = "Invalid Samples - " + get_last_dir_and_file_names(file_search)
        if show_samples:
            (non_valid,) = plt.plot(
                non_valid_optimization_obj_1[file_search],
                non_valid_optimization_obj_2[file_search],
                linestyle="None",
                marker=".",
                mew=0.5,
                markersize=3,
                fillstyle="none",
                label=label_is,
            )
            handler_map_for_legend[non_valid] = HandlerLine2D(numpoints=1)

    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(
            14
        )  # Set the fontsize of the label on the ticks of the x axis
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(
            14
        )  # Set the fontsize of the label on the ticks of the y axis

    # Add the legend with some customizations
    if print_legend:
        lgd = ax1.legend(
            handler_map=handler_map_for_legend,
            loc="best",
            bbox_to_anchor=(1, 1),
            fancybox=True,
            shadow=True,
            ncol=1,
            prop={"size": 14},
        )  # Display legend.

    font = {"size": 16}
    matplotlib.rc("font", **font)

    fig.savefig(output_image_pdf_file_with_all_samples, dpi=120, bbox_inches="tight")

    if objective_1_is_percentage:
        plt.xlim(0, 100)
    if objective_2_is_percentage:
        plt.ylim(0, 100)

    fig.savefig(output_image_pdf_file, dpi=120, bbox_inches="tight")


def compute_min_max_samples(input_data_array, x_max, x_min, xelem, y_max, y_min, yelem):
    """
    Compute the min and max on the x and y axis.

    :param input_data_array: computes the max and min on this data.
    :param x_max: input and output variable.
    :param x_min: input and output variable.
    :param xelem: variable to select the column that refers to the objective one in the array input_data_array.
    :param y_max: input and output variable.
    :param y_min: input and output variable.
    :param yelem: variable to select the column that refers to the objective two in the array input_data_array.
    :return: min and max on both axes
    """
    for elem in zip(input_data_array[xelem], input_data_array[yelem]):
        x_max = max(x_max, elem[0])
        y_max = max(y_max, elem[1])
        x_min = min(x_min, elem[0])
        y_min = min(y_min, elem[1])
    if x_min == float("inf"):
        print("Warning: x_min is infinity. Execution not interrupted.")
    if y_min == float("inf"):
        print("Warning: y_min is infinity. Execution not interrupted.")
    if x_max == float("-inf"):
        print("Warning: x_max is - infinity. Execution not interrupted.")
    if y_max == float("-inf"):
        print("Warning: y_max is - infinity. Execution not interrupted.")
    return x_max, x_min, y_max, y_min


def main():
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
        print("### Example 1: ")
        print("### hm-plot-dse example_scenarios/spatial/BlackScholes_scenario.json")
        print("### Example 2: ")
        print(
            "### hm-plot-dse example_scenarios/spatial/BlackScholes_scenario.json /path/to/output/image.pdf file1_pareto file1_search file2_pareto file2_search file3_pareto file3_search"
        )
        print("################################################")
        exit(1)

    plot(parameters_file, list_of_pairs_of_files, image_output_file)
    print("End of the plot_dse script!")


if __name__ == "__main__":
    main()
