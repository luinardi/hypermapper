import copy
import csv
import datetime
import os
import random
import sys
import warnings
from collections import OrderedDict

from jsonschema import exceptions

# ensure backward compatibility
try:
    from hypermapper import models
    from hypermapper import space
    from hypermapper.prior_optimization import prior_guided_optimization
    from hypermapper.random_scalarizations import random_scalarizations
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        are_all_elements_equal,
        concatenate_data_dictionaries,
        sample_weight_bbox,
        compute_data_array_scalarization,
        get_single_configuration,
        sample_weight_flat,
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

    from hypermapper import models
    from hypermapper import space
    from hypermapper.prior_optimization import prior_guided_optimization
    from hypermapper.random_scalarizations import random_scalarizations
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        are_all_elements_equal,
        concatenate_data_dictionaries,
        sample_weight_bbox,
        compute_data_array_scalarization,
        get_single_configuration,
        sample_weight_flat,
    )


def main(config, black_box_function=None, output_file="", profiling=None):
    """
    Run design-space exploration using bayesian optimization.
    :param config: dictionary containing all the configuration parameters of this optimization.
    :param output_file: a name for the file used to save the dse results.
    """
    start_time = datetime.datetime.now()
    run_directory = config["run_directory"]
    hypermapper_mode = config["hypermapper_mode"]["mode"]

    # Start logging
    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if hypermapper_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    # Log the json configuration for this optimization
    sys.stdout.write_to_logfile(str(config) + "\n")

    # Create parameter space object and unpack hyperparameters from json
    param_space = space.Space(config)
    application_name = config["application_name"]
    optimization_metrics = config["optimization_objectives"]
    optimization_iterations = config["optimization_iterations"]
    evaluations_per_optimization_iteration = config[
        "evaluations_per_optimization_iteration"
    ]
    batch_mode = evaluations_per_optimization_iteration > 1
    number_of_cpus = config["number_of_cpus"]
    print_importances = config["print_parameter_importance"]
    epsilon_greedy_threshold = config["epsilon_greedy_threshold"]
    acquisition_function = config["acquisition_function"]
    weight_sampling = config["weight_sampling"]
    scalarization_method = config["scalarization_method"]
    scalarization_key = config["scalarization_key"]
    doe_type = config["design_of_experiment"]["doe_type"]
    number_of_doe_samples = config["design_of_experiment"]["number_of_samples"]
    model_type = config["models"]["model"]
    optimization_method = config["optimization_method"]
    time_budget = config["time_budget"]
    input_params = param_space.get_input_parameters()
    number_of_objectives = len(optimization_metrics)
    objective_limits = {}
    data_array = {}
    fast_addressing_of_data_array = {}
    objective_bounds = None
    exhaustive_search_data_array = None
    normalize_objectives = False
    debug = False

    if "feasible_output" in config:
        feasible_output = config["feasible_output"]
        feasible_output_name = feasible_output["name"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
        enable_feasible_predictor_grid_search_on_recall_and_precision = feasible_output[
            "enable_feasible_predictor_grid_search_on_recall_and_precision"
        ]
        feasible_predictor_grid_search_validation_file = feasible_output[
            "feasible_predictor_grid_search_validation_file"
        ]
        feasible_parameter = param_space.get_feasible_parameter()
        number_of_trees = config["models"]["number_of_trees"]

    if weight_sampling == "bounding_box":
        objective_bounds = {}
        user_bounds = config["bounding_box_limits"]
        if len(user_bounds) == 2:
            if user_bounds[0] > user_bounds[1]:
                user_bounds[0], user_bounds[1] = user_bounds[1], user_bounds[0]
            for objective in optimization_metrics:
                objective_bounds[objective] = user_bounds
                objective_limits[objective] = user_bounds
        elif len(user_bounds) == number_of_objectives * 2:
            idx = 0
            for objective in optimization_metrics:
                objective_bounds[objective] = user_bounds[idx : idx + 2]
                if objective_bounds[objective][0] > objective_bounds[objective][1]:
                    objective_bounds[objective][0], objective_bounds[objective][1] = (
                        objective_bounds[objective][1],
                        objective_bounds[objective][0],
                    )
                objective_limits[objective] = objective_bounds[objective]
                idx += 2
        else:
            print(
                "Wrong number of bounding boxes, expected 2 or",
                2 * number_of_objectives,
                "got",
                len(user_bounds),
            )
            raise SystemExit
    else:
        for objective in optimization_metrics:
            objective_limits[objective] = [float("inf"), float("-inf")]

    if output_file == "":
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
    else:
        output_data_file = output_file

    exhaustive_search_data_array = None
    exhaustive_search_fast_addressing_of_data_array = None
    if hypermapper_mode == "exhaustive":
        exhaustive_file = config["hypermapper_mode"]["exhaustive_search_file"]
        (
            exhaustive_search_data_array,
            exhaustive_search_fast_addressing_of_data_array,
        ) = param_space.load_data_file(
            exhaustive_file, debug=False, number_of_cpus=number_of_cpus
        )

    # Check if some parameters are correctly defined
    if hypermapper_mode == "default":
        if black_box_function == None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    if (model_type == "gaussian_process") and (acquisition_function == "TS"):
        print(
            "Error: The TS acquisition function with Gaussian Process models is still under implementation"
        )
        print("Using EI acquisition function instead")
        config["acquisition_function"] = "EI"

    if number_of_cpus > 1:
        print(
            "Warning: HyperMapper supports only sequential execution for now. Running on a single cpu."
        )
        number_of_cpus = 1

    # If priors are present, use prior-guided optimization
    user_priors = False
    for input_param in config["input_parameters"]:
        if config["input_parameters"][input_param]["prior"] != "uniform":
            if number_of_objectives == 1:
                user_priors = True
            else:
                print(
                    "Warning: prior optimization does not work with multiple objectives yet, priors will be uniform"
                )
                config["input_parameters"][input_param]["prior"] = "uniform"

    if user_priors:
        bo_method = prior_guided_optimization
    else:
        bo_method = random_scalarizations
        normalize_objectives = True

    ### Resume previous optimization, if any
    beginning_of_time = param_space.current_milli_time()
    absolute_configuration_index = 0
    doe_t0 = datetime.datetime.now()
    if config["resume_optimization"] == True:
        resume_data_file = config["resume_optimization_data"]

        if not resume_data_file.endswith(".csv"):
            print("Error: resume data file must be a CSV")
            raise SystemExit
        if resume_data_file == "output_samples.csv":
            resume_data_file = application_name + "_" + resume_data_file

        data_array, fast_addressing_of_data_array = param_space.load_data_file(
            resume_data_file, debug=False, number_of_cpus=number_of_cpus
        )
        absolute_configuration_index = len(
            data_array[list(data_array.keys())[0]]
        )  # get the number of points evaluated in the previous run
        beginning_of_time = (
            beginning_of_time - data_array[param_space.get_timestamp_parameter()[0]][-1]
        )  # Set the timestamp back to match the previous run
        print(
            "Resumed optimization, number of samples = %d ......."
            % absolute_configuration_index
        )

    ### DoE phase
    if absolute_configuration_index < number_of_doe_samples:
        configurations = []
        default_configuration = param_space.get_default_or_random_configuration()
        str_data = param_space.get_unique_hash_string_from_values(default_configuration)
        if str_data not in fast_addressing_of_data_array:
            fast_addressing_of_data_array[str_data] = absolute_configuration_index
            configurations.append(default_configuration)
            absolute_configuration_index += 1

        doe_configurations = []
        if absolute_configuration_index < number_of_doe_samples:
            doe_configurations = param_space.get_doe_sample_configurations(
                fast_addressing_of_data_array,
                number_of_doe_samples - absolute_configuration_index,
                doe_type,
            )
        configurations += doe_configurations
        print(
            "Design of experiment phase, number of new doe samples = %d ......."
            % len(configurations)
        )

        doe_data_array = param_space.run_configurations(
            hypermapper_mode,
            configurations,
            beginning_of_time,
            black_box_function,
            exhaustive_search_data_array,
            exhaustive_search_fast_addressing_of_data_array,
            run_directory,
            batch_mode=batch_mode,
        )
        data_array = concatenate_data_dictionaries(
            data_array,
            doe_data_array,
            param_space.input_output_and_timestamp_parameter_names,
        )
        absolute_configuration_index = number_of_doe_samples
        iteration_number = 1
    else:
        iteration_number = absolute_configuration_index - number_of_doe_samples + 1

    # If we have feasibility constraints, we must ensure we have at least one feasible and one infeasible sample before starting optimization
    # If this is not true, continue design of experiment until the condition is met
    if enable_feasible_predictor:
        while (
            are_all_elements_equal(data_array[feasible_parameter[0]])
            and optimization_iterations > 0
        ):
            print(
                "Warning: all points are either valid or invalid, random sampling more configurations."
            )
            print("Number of doe samples so far:", absolute_configuration_index)
            configurations = param_space.get_doe_sample_configurations(
                fast_addressing_of_data_array, 1, "random sampling"
            )
            new_data_array = param_space.run_configurations(
                hypermapper_mode,
                configurations,
                beginning_of_time,
                black_box_function,
                exhaustive_search_data_array,
                exhaustive_search_fast_addressing_of_data_array,
                run_directory,
                batch_mode=batch_mode,
            )
            data_array = concatenate_data_dictionaries(
                new_data_array,
                data_array,
                param_space.input_output_and_timestamp_parameter_names,
            )
            absolute_configuration_index += 1
            optimization_iterations -= 1

    # Create output file with explored configurations from resumed run and DoE
    with open(
        deal_with_relative_and_absolute_path(run_directory, output_data_file), "w"
    ) as f:
        w = csv.writer(f)
        w.writerow(param_space.get_input_output_and_timestamp_parameters())
        tmp_list = [
            param_space.convert_types_to_string(j, data_array)
            for j in param_space.get_input_output_and_timestamp_parameters()
        ]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])

    for objective in optimization_metrics:
        lower_bound = min(objective_limits[objective][0], min(data_array[objective]))
        upper_bound = max(objective_limits[objective][1], max(data_array[objective]))
        objective_limits[objective] = [lower_bound, upper_bound]
    print(
        "\nEnd of doe/resume phase, the number of evaluated configurations is: %d\n"
        % absolute_configuration_index
    )
    sys.stdout.write_to_logfile(
        (
            "End of DoE - Time %10.4f sec\n"
            % ((datetime.datetime.now() - doe_t0).total_seconds())
        )
    )
    if doe_type == "grid_search" and optimization_iterations > 0:
        print(
            "Warning: DoE is grid search, setting number of optimization iterations to 0"
        )
        optimization_iterations = 0

    ### Main optimization loop
    bo_t0 = datetime.datetime.now()
    run_time = (datetime.datetime.now() - start_time).total_seconds() / 60
    # run_time / time_budget < 1 if budget > elapsed time or budget == -1
    if time_budget > 0:
        print(
            "starting optimization phase, limited to run for ", time_budget, " minutes"
        )
    elif time_budget == 0:
        print("Time budget cannot be zero. To not limit runtime set time_budget = -1")
        sys.exit()

    configurations = []
    evaluation_budget = optimization_iterations * evaluations_per_optimization_iteration
    iteration_number = 0
    evaluation_count = 0
    while evaluation_count < evaluation_budget and run_time / time_budget < 1:
        if evaluation_count % evaluations_per_optimization_iteration == 0:
            iteration_number += 1
            print("Starting optimization iteration", iteration_number)
            iteration_t0 = datetime.datetime.now()

        model_t0 = datetime.datetime.now()
        regression_models, _, _ = models.generate_mono_output_regression_models(
            data_array,
            param_space,
            input_params,
            optimization_metrics,
            1.00,
            config,
            model_type=model_type,
            number_of_cpus=number_of_cpus,
            print_importances=print_importances,
            normalize_objectives=normalize_objectives,
            objective_limits=objective_limits,
        )

        classification_model = None
        if enable_feasible_predictor:
            classification_model, _, _ = models.generate_classification_model(
                application_name,
                param_space,
                data_array,
                input_params,
                feasible_parameter,
                1.00,
                config,
                debug,
                number_of_cpus=number_of_cpus,
                data_array_exhaustive=exhaustive_search_data_array,
                enable_feasible_predictor_grid_search_on_recall_and_precision=enable_feasible_predictor_grid_search_on_recall_and_precision,
                feasible_predictor_grid_search_validation_file=feasible_predictor_grid_search_validation_file,
                print_importances=print_importances,
            )
        model_t1 = datetime.datetime.now()
        sys.stdout.write_to_logfile(
            (
                "Model fitting time %10.4f sec\n"
                % ((model_t1 - model_t0).total_seconds())
            )
        )
        if weight_sampling == "bounding_box":
            objective_weights = sample_weight_bbox(
                optimization_metrics, objective_bounds, objective_limits, 1
            )[0]
        elif weight_sampling == "flat":
            objective_weights = sample_weight_flat(optimization_metrics, 1)[0]
        else:
            print("Error: unrecognized option:", weight_sampling)
            raise SystemExit

        data_array_scalarization, _ = compute_data_array_scalarization(
            data_array, objective_weights, objective_limits, scalarization_method
        )
        data_array[scalarization_key] = data_array_scalarization.tolist()

        epsilon = random.uniform(0, 1)
        local_search_t0 = datetime.datetime.now()
        if epsilon > epsilon_greedy_threshold:
            best_configuration = bo_method(
                config,
                data_array,
                param_space,
                fast_addressing_of_data_array,
                regression_models,
                iteration_number,
                objective_weights,
                objective_limits,
                classification_model,
                profiling,
            )

        else:
            sys.stdout.write_to_logfile(
                str(epsilon)
                + " < "
                + str(epsilon_greedy_threshold)
                + " random sampling a configuration to run\n"
            )
            tmp_fast_addressing_of_data_array = copy.deepcopy(
                fast_addressing_of_data_array
            )
            best_configuration = (
                param_space.random_sample_configurations_without_repetitions(
                    tmp_fast_addressing_of_data_array, 1
                )[0]
            )
        local_search_t1 = datetime.datetime.now()
        sys.stdout.write_to_logfile(
            (
                "Local search time %10.4f sec\n"
                % ((local_search_t1 - local_search_t0).total_seconds())
            )
        )

        configurations.append(best_configuration)

        # When we have selected "evaluations_per_optimization_iteration" configurations, evaluate the batch
        if evaluation_count % evaluations_per_optimization_iteration == (
            evaluations_per_optimization_iteration - 1
        ):
            black_box_function_t0 = datetime.datetime.now()
            new_data_array = param_space.run_configurations(
                hypermapper_mode,
                configurations,
                beginning_of_time,
                black_box_function,
                exhaustive_search_data_array,
                exhaustive_search_fast_addressing_of_data_array,
                run_directory,
                batch_mode=batch_mode,
            )
            black_box_function_t1 = datetime.datetime.now()
            sys.stdout.write_to_logfile(
                (
                    "Black box function time %10.4f sec\n"
                    % ((black_box_function_t1 - black_box_function_t0).total_seconds())
                )
            )

            # If running batch BO, we will have some liars in fast_addressing_of_data, update them with the true value
            for configuration_idx in range(
                len(new_data_array[list(new_data_array.keys())[0]])
            ):
                configuration = get_single_configuration(
                    new_data_array, configuration_idx
                )
                str_data = param_space.get_unique_hash_string_from_values(configuration)
                if str_data in fast_addressing_of_data_array:
                    absolute_index = fast_addressing_of_data_array[str_data]
                    for header in configuration:
                        data_array[header][absolute_index] = configuration[header]
                else:
                    fast_addressing_of_data_array[
                        str_data
                    ] = absolute_configuration_index
                    absolute_configuration_index += 1
                    for header in configuration:
                        data_array[header].append(configuration[header])

            # and save results
            with open(
                deal_with_relative_and_absolute_path(run_directory, output_data_file),
                "a",
            ) as f:
                w = csv.writer(f)
                tmp_list = [
                    param_space.convert_types_to_string(j, new_data_array)
                    for j in list(
                        param_space.get_input_output_and_timestamp_parameters()
                    )
                ]
                tmp_list = list(zip(*tmp_list))
                for i in range(len(new_data_array[optimization_metrics[0]])):
                    w.writerow(tmp_list[i])
            configurations = []
        else:
            # If we have not selected all points in the batch yet, add the model prediction as a 'liar'
            for header in best_configuration:
                data_array[header].append(best_configuration[header])

            bufferx = [tuple(best_configuration.values())]
            prediction_means, _ = models.compute_model_mean_and_uncertainty(
                bufferx, regression_models, model_type, param_space
            )
            for objective in prediction_means:
                data_array[objective].append(prediction_means[objective][0])

            if classification_model is not None:
                classification_prediction_results = models.model_probabilities(
                    bufferx, classification_model, param_space
                )
                true_value_index = (
                    classification_model[feasible_parameter[0]]
                    .classes_.tolist()
                    .index(True)
                )
                feasibility_indicator = classification_prediction_results[
                    feasible_parameter[0]
                ][:, true_value_index]
                data_array[feasible_output_name].append(
                    True if feasibility_indicator[0] >= 0.5 else False
                )

            data_array[param_space.get_timestamp_parameter()[0]].append(
                absolute_configuration_index
            )
            str_data = param_space.get_unique_hash_string_from_values(
                best_configuration
            )
            fast_addressing_of_data_array[str_data] = absolute_configuration_index
            absolute_configuration_index += 1

        for objective in optimization_metrics:
            lower_bound = min(
                objective_limits[objective][0], min(data_array[objective])
            )
            upper_bound = max(
                objective_limits[objective][1], max(data_array[objective])
            )
            objective_limits[objective] = [lower_bound, upper_bound]

        evaluation_count += 1
        run_time = (datetime.datetime.now() - start_time).total_seconds() / 60
        iteration_t1 = datetime.datetime.now()
        sys.stdout.write_to_logfile(
            (
                "Total iteration time %10.4f sec\n"
                % ((iteration_t1 - iteration_t0).total_seconds())
            )
        )

        if profiling is not None:
            profiling.add("Model fitting time", (model_t1 - model_t0).total_seconds())
            # local search profiling is done inside of local search
            profiling.add(
                "Black box function time",
                (black_box_function_t1 - black_box_function_t0).total_seconds(),
            )

    sys.stdout.write_to_logfile(
        (
            "End of BO phase - Time %10.4f sec\n"
            % ((datetime.datetime.now() - bo_t0).total_seconds())
        )
    )

    print("End of Bayesian Optimization")

    print_posterior_best = config["print_posterior_best"]
    if print_posterior_best:
        if number_of_objectives > 1:
            print(
                "Warning: print_posterior_best is set to true, but application is not mono-objective."
            )
            print(
                "Can only compute best according to posterior for mono-objective applications. Ignoring."
            )
        elif enable_feasible_predictor:
            print(
                "Warning: print_posterior_best is set to true, but application has feasibility constraints."
            )
            print(
                "Cannot compute best according to posterior for applications with feasibility constraints. Ignoring."
            )
        else:
            # Update model with latest data
            regression_models, _, _ = models.generate_mono_output_regression_models(
                data_array,
                param_space,
                input_params,
                optimization_metrics,
                1.00,
                config,
                model_type=model_type,
                number_of_cpus=number_of_cpus,
                print_importances=print_importances,
                normalize_objectives=normalize_objectives,
                objective_limits=objective_limits,
            )

            best_point = models.minimize_posterior_mean(
                regression_models,
                config,
                param_space,
                data_array,
                objective_limits,
                normalize_objectives,
                profiling,
            )
            keys = ""
            best_point_string = ""
            for key in best_point:
                keys += f"{key},"
                best_point_string += f"{best_point[key]},"
            keys = keys[:-1]
            best_point_string = best_point_string[:-1]

            sys.stdout.write_protocol("Minimum of the posterior mean:\n")
            sys.stdout.write_protocol(f"{keys}\n")
            sys.stdout.write_protocol(f"{best_point_string}\n\n")

    sys.stdout.write_to_logfile(
        (
            "Total script time %10.2f sec\n"
            % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )

    return data_array
