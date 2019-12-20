# This script computes the Pareto from a csv file. It outputs a csv file containing the Pareto.
import sys
import space
import datetime
import numpy as np
import csv
import json
from jsonschema import Draft4Validator
import os
from utility_functions import *

def sequential_is_pareto_efficient_dumb(costs):
    """ This function is in general more efficient for high-dimensional Pareto fronts. 
        Use sequential_is_pareto_efficient(costs) for low-dimensional Pareto fronts
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    isString = isinstance(costs[0][0], str) # Not sure this works for python 3 (for python 3 this should work: isinstance(s, str))
    if isString :
        costs = costs.astype(np.float)
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs>=c, axis=1))
    # This part cleans up, removing points that have one metric that is equal while the second metric is different (and worse than the first one)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if np.any(np.logical_and(np.any(costs[is_efficient] == c, axis=1), np.any(costs[is_efficient] < c, axis=1))) :
                is_efficient[i] = False
    return is_efficient


## I should tweak this function to include the x_operator, y_operator.
################# Test this function using:
#rng = np.random.RandomState(1234)
#costs = rng.rand(5, 2)
#less_dumb__ixs = sequential_is_pareto_efficient(costs)
#################
def sequential_is_pareto_efficient(costs):
    """ This function is in general more efficient for low-dimensional Pareto fronts. 
        Use sequential_is_pareto_efficient_dumb(costs) for high-dimensional Pareto fronts
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    isString = isinstance(costs[0][0], str)
    if isString : 
        #costs = costs.astype(np.int) # It doesn't seem that this speeds the conversion up by any good factor
        costs = costs.astype(np.float)
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            #is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
            tmp1 = costs[is_efficient]<=c
            tmp = np.any(tmp1, axis=1)
            is_efficient[is_efficient] = tmp

    # This part cleans up, removing points that have one metric that is equal while the second metric is different (and worse than the first one)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if np.any(np.logical_and(np.any(costs[is_efficient] == c, axis=1), np.any(costs[is_efficient] < c, axis=1))) :
                is_efficient[i] = False

    # This part cleans up, removing points that have one metric that is equal while the second metric is different (and worse than the first one)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             tmp1 = costs[is_efficient] == c
#             tmp2 = costs[is_efficient] < c
#             tmp1_any = np.any(tmp1, axis=1)
#             tmp2_any = np.any(tmp2, axis=1)
#             tmp = np.logical_and(tmp1_any, tmp2_any)
#             if np.any(tmp) :
#                 is_efficient[i] = False

    return is_efficient

def compute_pareto_lines(data_array, data_size, x_select, x_operator, y_select, y_operator, filter_variable = None, filter_value = None, filter_function = True) :
    """
    This is the original version, a reimplementation that may have slightly different signature is sequential_is_pareto_efficient().
    This function computes Pareto curves from data_array.
    It may be important to set the x_operator and y_operator.
    You can also find useful the filter parameter.
    :param data_array:
    :param data_size:
    :param x_select:
    :param x_operator:
    :param y_select:
    :param y_operator:
    :param filter_variable:
    :param filter_value:
    :param filter_function:
    :return:
    """
    paretolines = []
    if filter_variable == None :
        filtered_list =  list(range(data_size))
    else :
        filtered_list = [x for x in range(data_size) if data_array[filter_variable][x] == filter_value and (filter_function(data_array,x))  ]
    To_test = filtered_list[:]

    for i in filtered_list :
        xval = float(data_array[x_select][i])
        yval = float(data_array[y_select][i])
        strong = True
        for j in To_test :
            xvalbis = float(data_array[x_select][j])
            yvalbis = float(data_array[y_select][j])
            if  (x_operator (xvalbis , xval) and y_operator (yvalbis , yval)) or ((xvalbis == xval) and y_operator (yvalbis , yval)) or ((yvalbis == yval) and x_operator (xvalbis , xval)) :
                strong = False
                break
        if strong :
            paretolines.append(i)
        else :
            To_test.remove(i)
    return paretolines

def parallel_is_pareto_efficient(debug, predictions, costs, number_of_cpus=0):
    """
    Return the Pareto of predictions in a new array predictions.
    :param debug:
    :param predictions:
    :param costs:
    :param number_of_cpus:
    :return:
    """
    only_keep_concatenated = domain_decomposition_and_parallel_computation(debug, sequential_is_pareto_efficient, np.concatenate, costs, number_of_cpus)
    
    for p_k in predictions:
        predictions[p_k] = predictions[p_k][only_keep_concatenated]

    # We need to apply the Pareto again with the results of the parallel Pareto computation. 
    #The single Pareto computations give Paretos for those subset and now we are computing the final aggregated Pareto (this time the input array will be much smaller).
    costs_reduction = costs[only_keep_concatenated]
    only_keep = sequential_is_pareto_efficient(costs_reduction)
    return only_keep

def compute_pareto(param_space, input_data_file, output_pareto_file, 
                    debug=False, number_of_cpus=0):
    """
    This function computes a Pareto from a csv file called input_data_file.
    The Pareto is saved in the output_pareto_file.
    It may probably be accelerated if needed.
    :param param_space: Space object defined in the json schema.
    :param input_data_file: name of the file containing the DSE data.
    :param output_pareto_file: name of the file where the output Pareto samples are saved.
    :param debug: whether to print debugging information
    :param number_of_cpus: not used yet (for future dev).
    :return:
    """
    optimization_metrics = param_space.get_optimization_parameters()
    x_select = optimization_metrics[0]
    y_select = optimization_metrics[1]
    feasible = param_space.get_feasible_parameter()[0] # returns a list, we just want the name
    count_number_of_points_in_Pareto = 0

    data_array, fast_addressing_of_data_array = param_space.load_data_file(input_data_file, debug, number_of_cpus=number_of_cpus)

    if debug: # When debugging we want to know how many rows are invalid before filtering
        count = 0
        for i in range(len(data_array[feasible])):
            if data_array[feasible][i] == False:
                count += 1
        print(("Number of false before filtering " + str(count)))

    # Check if feasibility flag was set, if so, remove non valid rows before computing the Pareto
    if feasible is not None:
        i = 0
        for ind in range(len(data_array[feasible])):
            if data_array[feasible][i] == False:
                for key in list(data_array.keys()):
                    del data_array[key][i]
            else:
                i += 1

        if len(data_array[feasible]) == 0:
            print("Warning: after removing the non-valid rows in file %s the data array is now empty."%input_data_file)
            with open(output_pareto_file, 'w') as f:
                w = csv.writer(f)
                w.writerow(list(data_array.keys()))
            return count_number_of_points_in_Pareto

        if debug: # When debugging we want to know how many rows are invalid after filtering
            count = 0
            for i in range(len(data_array[feasible])):
                if data_array[feasible][i] == False:
                    count += 1
            print(("Number of false after filtering " + str(count)))

    costs = np.column_stack((data_array[x_select], data_array[y_select]))
    bool_indicator_paretoline = sequential_is_pareto_efficient(costs)

    # Write on file the Pareto
    with open(output_pareto_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(list(data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, data_array) for j in list(data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(bool_indicator_paretoline)):
            if bool_indicator_paretoline[i]:
                w.writerow(tmp_list[i])
                count_number_of_points_in_Pareto += 1

    return count_number_of_points_in_Pareto


def main(parameters_file="example_scenarios/spatial/BlackScholes_scenario.json", input_data_file=None, output_pareto_file=None) :
    """
    Compute Pareto from the csv data files specified in the json output_pareto_file field.
    :param parameters_file: the json file the specify all the HyperMapper input parameters.
    :return: the csv file is written on disk.
    """

    print("######## compute_pareto.py #####################");
    print("### Parameters file is %s" % parameters_file); sys.stdout.flush()

    filename, file_extension = os.path.splitext(parameters_file)
    if file_extension != ".json":
        print("Error: invalid file name. \nThe input file has to be a .json file not a %s" % file_extension)
        exit(1)
    with open(parameters_file, 'r') as f:
        config = json.load(f)

    json_schema_file = 'scripts/schema.json'
    with open(json_schema_file, 'r') as f:
        schema = json.load(f)

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    DefaultValidatingDraft4Validator(schema).validate(config)

    application_name = config["application_name"]
    max_number_of_predictions = config['max_number_of_predictions']
    optimization_metrics = config["optimization_objectives"]
    number_of_cpus = config["number_of_cpus"]
    run_directory = config["run_directory"]
    if input_data_file == None:
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
        input_data_file = deal_with_relative_and_absolute_path(run_directory, output_data_file)
    if output_pareto_file == None:
        output_pareto_file = config["output_pareto_file"]
        if output_pareto_file == "output_pareto.csv":
            output_pareto_file = application_name + "_" + output_pareto_file
        output_pareto_file = deal_with_relative_and_absolute_path(run_directory, output_pareto_file)

    param_space = space.Space(config)
    print("### The input data file is %s" % input_data_file)
    print("### The output Pareto file is %s" % output_pareto_file)
    print("################################################")
    debug = False

    print("Computing the Pareto...")
    start_time = datetime.datetime.now()
    # Compute Pareto and save it to output_pareto_file
    with open(input_data_file, 'r') as f_csv_file_data_array:
        count_number_of_points_in_Pareto = compute_pareto(param_space, input_data_file, output_pareto_file,
                                                          debug, number_of_cpus)
    end_time = datetime.datetime.now()
    print(("Total time of computation is (read and Pareto computation): " + str((end_time - start_time).total_seconds()) + " seconds"))
    print(("The total size of the Pareto (RS + AL) is: %d" % count_number_of_points_in_Pareto)); sys.stdout.flush()
    print("End of the compute_pareto.py script!\n")


if __name__ == "__main__":

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
    else :
        print("Error: only one argument needed, the parameters json file.")

    if parameters_file == "--help" or len(sys.argv) < 2 or len(sys.argv) >= 5:
        print("################################################")
        print("### cd hypermapper")
        print("### Example 1: ")
        print("### python3 scripts/compute_pareto.py example_scenarios/spatial/app_scenario.json")
        print("### Example 2: ")
        print("### python3 scripts/compute_pareto.py example_scenarios/spatial/app_scenario.json /path/to/input_data_file.csv /path/to/output_pareto_file.csv ")
        print("################################################")
        exit(1)

    main(parameters_file, input_data_file, output_pareto_file)
