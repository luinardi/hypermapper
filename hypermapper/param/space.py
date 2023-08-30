import copy
import csv
import itertools
import re
import sys
import time
from typing import Any, Dict, List, Tuple, Callable, Optional

import networkx as nx
import numpy as np
import torch

import hypermapper.param.constraints as constraints
from hypermapper.param.chain_of_trees import ChainOfTrees, Node, Tree
from hypermapper.param.data import DataArray
from hypermapper.param.parameters import (
    RealParameter,
    IntegerParameter,
    OrdinalParameter,
    CategoricalParameter,
    PermutationParameter,
)


class Space:
    def __init__(self, settings: Dict):
        """
        Space is the class the defines the (input) search space.
        The type and sizes of input parameters are kept here.

        The main data format is in three torch tensors, X(nxm), Y(nxp), F(n) containing the configurations, the values and the feasibility respectively.
        Additionally, there is time_stamps which contains the evaluation timestamps for each evaluation.

        The configurations X are stored in their "internal form" where categoricals and permutations are stored as numeric indices. There are three other
        representations, "String representation" used to print and check constraints, "original representation", which is used for evaluating constraints, and a
        [0-1] representation. Changing between the representations is done by the convert function.

        Space also has a number of utility functions for manipulating the data.

        Input:
            - settings: the json settings.
        """
        self.settings = settings
        self.application_name = settings["application_name"]
        self.parameters = []
        self.parameter_types = []
        self.parameter_python_types = []
        self.metric_names = settings["optimization_objectives"]
        if ("feasible_output" in settings) and settings["feasible_output"][
            "enable_feasible_predictor"
        ] is True:
            self.enable_feasible_predictor = True
            feasible_output = settings["feasible_output"]
            self.feasible_output_name = feasible_output["name"]
            self.true_value = feasible_output["true_value"]
            self.false_value = feasible_output["false_value"]
        else:
            self.feasible_output_name = None
            self.enable_feasible_predictor = False

        # Process input parameters from the json file
        self.number_parameters = []
        self.categorical_parameters = []
        self.permutation_parameters = []
        self.real_parameters = []
        self.conditional_space = False
        self.constraints = []

        self.parse_input_parameters(settings["input_parameters"])

        self.parameter_names = [parameter.name for parameter in self.parameters]
        self.parameter_indices = {
            parameter_name: i for i, parameter_name in enumerate(self.parameter_names)
        }
        self.dimension = len(self.parameters)
        self.has_real_parameters = "real" in self.parameter_types
        self.only_real_parameters = len(self.real_parameters) == self.dimension
        if self.has_real_parameters:
            self.size = np.inf
        else:
            self.size = 1
            for parameter in self.parameters:
                self.size *= parameter.get_size()
        self.input_output_parameter_names = (
            self.parameter_names
            + self.metric_names
            + ([self.feasible_output_name] if self.enable_feasible_predictor else [])
        )
        self.all_names = self.input_output_parameter_names + ["timestamp"]
        self.normalize_inputs = settings["normalize_inputs"]
        self.normalize_priors = True

        if self.conditional_space:
            cot_order, tree_orders = self.get_cot_order()
            self.chain_of_trees = ChainOfTrees(
                cot_order, len(cot_order) == self.dimension
            )
            self.create_chain_of_trees(tree_orders)

            # cot_parameters are the constraints which are part of the chain of trees. Floats and variables which have dependencies with float parameters are not.
            self.cot_parameters = [
                self.parameters[i] for i in range(self.dimension) if i in cot_order
            ]
            self.non_cot_parameters = [
                self.parameters[i] for i in range(self.dimension) if i not in cot_order
            ]
            cot_ordered_names = [p.name for p in self.cot_parameters] + [
                p.name for p in self.non_cot_parameters
            ]
            self.cot_remap = [cot_ordered_names.index(p.name) for p in self.parameters]
            self.non_cot_constraints = []
            for p in self.non_cot_parameters:
                if p.constraints is not None:
                    self.non_cot_constraints.extend(p.constraints)

        self.use_gradient_descent = (
            settings["models"]["model"] == "gaussian_process"
            and not self.enable_feasible_predictor
            and not self.conditional_space
            and not settings["GP_model"] == "gpy"
            and self.has_real_parameters
        )

    def parse_input_parameters(self, input_parameters: Dict):
        """
        Parse the input search space from the json file.

        Input:
            - input_parameters: A dict with the parameter information from the json settings file.
        """
        for param_name, param in input_parameters.items():
            param_type = param["parameter_type"]
            param_default = None
            if "parameter_default" in param:
                param_default = param["parameter_default"]
            param_distribution = param["prior"]
            if param_distribution in ["gaussian"]:
                param_distribution = (param_distribution, param["prior_parameters"])

            param_constraints = None
            if "constraints" in param:
                self.conditional_space = True
                param_constraints = param["constraints"]
                self.constraints.extend(param_constraints)

            param_dependencies = None
            if "dependencies" in param:
                self.conditional_space = True
                param_dependencies = param["dependencies"]

            if "transform" in param:
                transform = param["transform"]
            else:
                transform = "none"

            if param_type == "real":
                param_min, param_max = param["values"]
                param_discretization = np.linspace(param_min, param_max, num=10)
                param_obj = RealParameter(
                    name=param_name,
                    min_value=param_min,
                    max_value=param_max,
                    default=param_default,
                    preferred_discretization=param_discretization,
                    probability_distribution=param_distribution,
                    constraints=param_constraints,
                    dependencies=param_dependencies,
                    transform=transform,
                )
                self.parameters.append(param_obj)
                self.number_parameters.append(param_obj)
                self.real_parameters.append(param_obj)
                self.parameter_types.append("real")
                self.parameter_python_types.append("float")

            elif param_type == "integer":
                param_min, param_max = param["values"]
                if param_min > param_max:
                    param_min, param_max = (
                        param_max,
                        param_min,
                    )  # breaks if min is greater than max. It does not break for real variables.
                param_obj = IntegerParameter(
                    name=param_name,
                    min_value=param_min,
                    max_value=param_max,
                    default=param_default,
                    probability_distribution=param_distribution,
                    constraints=param_constraints,
                    dependencies=param_dependencies,
                    transform=transform,
                )
                self.parameters.append(param_obj)
                self.number_parameters.append(param_obj)
                self.parameter_types.append("integer")
                self.parameter_python_types.append("int")

            elif param_type == "ordinal":
                param_values = param["values"]
                param_obj = OrdinalParameter(
                    name=param_name,
                    values=[np.float64(p) for p in param_values],
                    default=param_default,
                    probability_distribution=param_distribution,
                    constraints=param_constraints,
                    dependencies=param_dependencies,
                    transform=transform,
                )
                self.parameters.append(param_obj)
                self.number_parameters.append(param_obj)
                self.parameter_types.append("ordinal")
                self.parameter_python_types.append("float")

            elif param_type == "categorical":
                param_values = param["values"]
                param_obj = CategoricalParameter(
                    name=param_name,
                    values=param_values,
                    default=param_default,
                    probability_distribution=param_distribution,
                    constraints=param_constraints,
                    dependencies=param_dependencies,
                )
                self.parameters.append(param_obj)
                self.categorical_parameters.append(param_obj)
                self.parameter_types.append("categorical")
                self.parameter_python_types.append("int")

            elif param_type == "permutation":
                n_elements = param["values"][0]
                parametrization = param["parametrization"]
                param_obj = PermutationParameter(
                    name=param_name,
                    n_elements=n_elements,
                    default=param_default,
                    parametrization=parametrization,
                    constraints=param_constraints,
                    dependencies=param_dependencies,
                )
                self.parameters.append(param_obj)
                self.permutation_parameters.append(param_obj)
                self.parameter_types.append("permutation")
                self.parameter_python_types.append("int")

    def create_chain_of_trees(self, tree_orders: List[List[int]]):
        """
        Instantiates a chain of trees

        Input:
            - dependency_graph: A nx-graph describing the constraint dependencies between variables
        """
        for tree_order in tree_orders:
            tree = Tree()
            node = tree.get_root()
            build_time_start = time.time()
            self.build_tree(
                tree,
                tree.get_root(),
                torch.Tensor([[float("nan")] * len(tree_order)]),
                tree_order,
                -1,
            )
            sys.stdout.write_to_logfile(
                f"time to build tree: {time.time() - build_time_start}\n"
            )
            sys.stdout.write_to_logfile(f"number of leaves: {len(tree.leaves)}\n")
            propagate_probabilities_time_start = time.time()
            node.propagate_probabilities(self)
            sys.stdout.write_to_logfile(
                f"time to calculate probabilities: {time.time() - propagate_probabilities_time_start}\n"
            )
            node.set_children_dict()
            tree.depth = len(tree_order)
            if len(tree.get_leaves()) > 0:
                self.chain_of_trees.add_tree(tree)

    def build_tree(
        self,
        tree: Tree,
        node: Node,
        configuration: torch.Tensor(),
        tree_order: List[int],
        level: int,
    ):
        """
        Builds a subtree rooted in 'node' with all feasible solutions given the partial configuration provided in 'configuration'
        NOTE: configurations here are not ordered in the usual order but in the tree_order
        """
        child_parameter = self.parameters[tree_order[level + 1]]
        tree_ordered_parameters = [self.parameters[p] for p in tree_order[: level + 1]]
        possible_child_values = constraints.filter_conditional_values(
            child_parameter,
            child_parameter.constraints,
            {
                p.name: v
                for p, v in zip(
                    tree_ordered_parameters,
                    self.convert(
                        configuration[:, : level + 1],
                        "internal",
                        "original",
                        parameters=tree_ordered_parameters,
                    )[0],
                )
                if not np.isnan(v)
            },
        )
        if len(possible_child_values) > 0:
            if level == len(tree_order) - 2:
                for idx, child_value in enumerate(possible_child_values):
                    child_node = Node(
                        node,
                        child_value,
                        child_parameter.name,
                        child_parameter.get_index(child_value.item()),
                    )
                    node.add_child(child_node)
                    tree.add_leaf(child_node)
            else:
                for idx, child_value in enumerate(possible_child_values):
                    child_configuration = copy.deepcopy(configuration)
                    child_configuration[0, level + 1] = child_value
                    child_node = Node(
                        node,
                        child_value,
                        child_parameter.name,
                        child_parameter.get_index(child_value.item()),
                    )
                    self.build_tree(
                        tree, child_node, child_configuration, tree_order, level + 1
                    )
                    if child_node.children:
                        node.add_child(child_node)

    def get_cot_order(self) -> Tuple[List[int], List[List[int]]]:
        """
        Creates dependency graph to find a topological sorting of the parameters.
        """
        dependency_graph = nx.DiGraph()
        for parameter in self.parameters:
            if not isinstance(parameter, RealParameter):
                dependencies = parameter.get_dependencies()
                if dependencies is None:
                    dependency_graph.add_node(parameter.name)
                else:
                    for dependency in dependencies:
                        if not isinstance(
                            self.parameters[self.parameter_names.index(dependency)],
                            RealParameter,
                        ):
                            dependency_graph.add_edge(dependency, parameter.name)

        topological_order = list(nx.topological_sort(dependency_graph))
        subgraphs = nx.connected_components(dependency_graph.to_undirected())
        cot_order = []
        tree_orders = []
        for subgraph in subgraphs:
            cot_order += [
                self.parameter_names.index(p)
                for p in topological_order
                if p in subgraph
            ]
            tree_orders.append(
                [
                    self.parameter_names.index(p)
                    for p in topological_order
                    if p in subgraph
                ]
            )
        return cot_order, tree_orders

    def get_space(self) -> torch.Tensor:
        # update
        """
        Get the Cartesian product of the discrete parameters of input previously declared in the json file.
        Warning: use this function only if the space size is small (how small will depend on the machine used,
        perhaps smaller than 10,000,000 in any case). Can only be used in discrete spaces.
        Returns:
            - a list that contains each configuration as a dictionary.
        """
        itertool_cartesian_product = itertools.product(
            *[parameter.get_values() for parameter in self.parameters]
        )
        return torch.tensor(
            [configuration for configuration in itertool_cartesian_product]
        )

    def get_default_configuration(self):
        """
        Returns:
            - the default configuration (internal) from the input parameters space if complete and feasible.
        """
        default_configuration = [p.default for p in self.parameters]
        if None in default_configuration:
            sys.stdout.write_to_logfile("Warning: incomplete default")
            return None
        default_configuration = torch.tensor([default_configuration])
        if self.conditional_space and not self.evaluate(default_configuration):
            sys.stdout.write_to_logfile("Warning: the default configuration is infeasible. Are you sure you want this?.")
            return default_configuration
        return default_configuration

    def convert(
        self,
        data: Any,
        from_type: str,
        to_type: str,
        parameters=None,
    ) -> Any:
        """
        Converts between data representations.

        Input:
            - data: the data to convert.
            - from_type: the data format.
            - to_type: the output format.
        Returns:
            - the converted data.

        Available forms are:
            - "internal" (default) : torch.Tensor
            - "string" : List[List[String]]
            - "original" : List[List[Any]]
            - "01" : torch.tensor
        """

        if parameters is None:
            parameters = self.parameters

        if from_type in ["internal", "01"]:
            n_rows, n_columns = data.shape
            data = data.numpy()
        elif from_type in ["string", "original"]:
            n_rows = len(data)
            n_columns = len(data[0])
        else:
            raise Exception("Invalid from_type", from_type)
        rows = range(n_rows)
        columns = range(n_columns)
        output = [
            [
                parameters[column].convert(data[row][column], from_type, to_type)
                for column in columns
            ]
            for row in rows
        ]
        if to_type in ["internal", "01"]:
            output = torch.tensor(output, dtype=torch.float64)

        return output

    def evaluate(self, configurations: torch.Tensor, CoT=False) -> List[bool]:
        """
        Evaluates complete configurations
        Input:
            - configurations: configurations to evaluate ("internal" representation)
        Return:
            - List of booleans
        """

        if CoT and not self.has_real_parameters:
            return self.evaluate_CoT(configurations)

        # transform the torch tensor to dict of lists - needed for the numexpr package
        transformed_configurations = {
            parameter_name: parameter_values
            for parameter_name, parameter_values in zip(
                self.parameter_names,
                [
                    list(i)
                    for i in zip(*self.convert(configurations, "internal", "original"))
                ],
            )
        }
        feasible = constraints.evaluate_constraints(
            self.constraints, transformed_configurations
        )

        return feasible

    def evaluate_CoT(self, configurations: torch.Tensor) -> List[bool]:
        """
        Evaluates a configuration by checking if the corresponding leaf exists in the tree. This is much faster than evaluating the constraint anew.
        Input:
            - configurations: configurations to evaluate ("internal" representation)
        Return:
            - List of booleans
        """

        feasible_list = []
        for configuration in configurations:
            cot_ordered_configuration = self.chain_of_trees.to_cot_order(configuration)
            feasible = True
            idx = 0
            for tree in self.chain_of_trees.trees:
                node = tree.get_root()
                for level in range(tree.depth):
                    value = cot_ordered_configuration[idx].item()
                    idx += 1
                    if value in node.children_dict:
                        node = node.children_dict[value]
                    else:
                        feasible = False
                        break
                if not feasible:
                    break
            feasible_list.append(feasible)
        return feasible_list

    def filter_out_previously_run(
        self,
        to_filter: torch.Tensor,
        previously_run: Dict[str, int],
    ) -> torch.Tensor:
        """
        Remove previously run configurations from a tensor.
        """
        accepted_indices = [
            i
            for i, l in enumerate(
                [self.get_unique_hash_string_from_values(x) for x in to_filter]
            )
            if l not in previously_run
        ]
        return to_filter[accepted_indices, :]

    def conditional_space_exhaustive_search(self):
        return self.chain_of_trees.get_all_configurations()

    def run_configurations(
        self,
        configurations: torch.Tensor,
        beginning_of_time: int,
        settings: Dict,
        black_box_function: Optional[Callable] = None,
    ):
        """
        Run a set of configurations in one of Hypermappers modes. Each time a configuration is run, it is also saved to the output file.

        Input:
            - configurations: a list of configurations (dict).
            - beginning_of_time: time from the beginning of the Hypermapper design space exploration.
            - settings:
            - black_box_function: objective function being optimized.

        Returns:
            - dictionary with the new configurations that were evaluated
        """
        if settings["hypermapper_mode"]["mode"] == "default":
            data_array = self.run_configurations_with_black_box_function(
                configurations,
                black_box_function,
                beginning_of_time,
                settings["output_data_file"],
            )
            self.print_data_array(data_array)

        elif settings["hypermapper_mode"]["mode"] == "client-server":
            print("Running on client-server mode.")
            data_array = self.run_configurations_client_server(
                configurations, settings["output_data_file"]
            )

        if any(torch.isnan(data_array.metrics_array)) or any(
            torch.isnan(data_array.metrics_array)
        ):
            raise Exception("NaN values in output from blackbox function. Exiting.")

        return data_array

    def run_configurations_client_server(
        self,
        configurations: torch.Tensor,
        output_data_file: str,
    ):
        """
        Run a set of configurations in client-server mode under the form of a list of configurations.

        Input:
            - configurations: a list of configurations (dict).
            - output_data_file: path to the output file.
        """
        print("Communication protocol: sending message...")

        # Write to stdout
        sys.stdout.write_protocol(
            "Request %d\n" % len(configurations)
        )  # From the Hypermapper/interacting system protocol
        str_header = ""
        for parameter_name in list(self.parameter_names):
            str_header += str(parameter_name) + ","
        str_header = str_header[:-1] + "\n"
        sys.stdout.write_protocol(str_header)

        str_configurations = self.convert(configurations, "internal", "string")
        for idx, configuration in enumerate(str_configurations):
            sys.stdout.write_protocol(",".join(configuration) + "\n")

        # Read from stdin
        print("Communication protocol: receiving message....")
        line = sys.stdin.readline()
        sys.stdout.write(line)
        parameters_header = [x.strip() for x in line.split(",")]
        parameters_index_reference = {}
        for header in self.input_output_parameter_names:
            if header not in parameters_header:
                print("Key Error while getting the random configurations results.")
                print("The key Hypermapper was looking for is: %s" % str(header))
                print("The headers received are: %s" % str(parameters_header))
                print(
                    "The input-ouput parameters specified in the json are: %s"
                    % str(self.all_names)
                )
                exit()
            parameters_index_reference[header] = parameters_header.index(header)
        parameters_data = []
        for conf_index in range(configurations.shape[0]):
            line = sys.stdin.readline()
            sys.stdout.write(line)
            # same as .split(','') but don't split on commas within parenthesis to allow for permutations
            parameters_data.append(
                [x.strip() for x in re.split(r",(?![^(]*[)])", line)]
            )
        try:
            metric_indices = [
                parameters_header.index(name) for name in self.metric_names
            ]
            param_indices = [
                parameters_header.index(name) for name in self.parameter_names
            ]
            new_configurations = []
            metrics = []
            feasible = []
            for row in parameters_data:
                new_configurations.append([row[p] for p in param_indices])
                metrics.append([float(row[m]) for m in metric_indices])
                if self.enable_feasible_predictor:
                    feasible.append(
                        (
                            True
                            if row[parameters_header.index(self.feasible_output_name)]
                            == str(self.true_value)
                            else False
                        )
                    )
            metrics = torch.tensor(metrics)
            feasible = torch.tensor(feasible)
        except ValueError as ve:
            print("Failed to parse received message:")
            print(ve)
            raise SystemError

        new_configurations = self.convert(
            new_configurations, "string", "internal"
        )
        timestamps = torch.tensor([self.current_milli_time()] * len(configurations))
        new_data_array = DataArray(new_configurations, metrics, timestamps, feasible)
        write_data_array(self, new_data_array, output_data_file)
        return new_data_array

    def run_configurations_with_black_box_function(
        self,
        configurations: torch.Tensor,
        black_box_function: Callable,
        beginning_of_time: float,
        output_data_file: str,
    ) -> DataArray:
        """
        Run a list of configurations.

        Input:
            - configurations: a list of configurations (dict).
            - black_box_function: objective function being optimized.
            - beginning_of_time: time from the beginning of the Hypermapper design space exploration.
            - output_data_file: path to the output file.
        """
        original_configurations = self.convert(configurations, "internal", "original")
        objective_values = []
        timestamps = []
        for original_configuration in original_configurations:
            output = black_box_function(
                {
                    name: value
                    for name, value in zip(self.parameter_names, original_configuration)
                }
            )
            if isinstance(output, tuple):
                output = list(output)
            if not (type(output) is list or type(output) is dict):
                output = [output]
            objective_values.append(output)
            timestamps.append(self.current_milli_time() - beginning_of_time)

        output_names = (
            self.metric_names
            + ([self.feasible_output_name] if self.enable_feasible_predictor else [])
            + (
                ["std_estimate"]
                if self.settings["GP_model"]
                in ["botorch_fixed", "botorch_heteroskedastic"]
                else []
            )
        )

        if type(objective_values[0]) is dict:
            for m in objective_values[0].keys():
                if m not in output_names:
                    raise Exception(
                        f"The black box function return metric {m} which is not in the json file."
                    )
            for m in output_names:
                if m not in objective_values[0].keys():
                    raise Exception(
                        f"The black box function should return metric {m} but didn't."
                    )
            objective_values = [
                [objective_dict[metric] for metric in output_names]
                for objective_dict in objective_values
            ]
        else:
            if len(objective_values[0]) != len(output_names):
                raise Exception(
                    f"The black box function should return {len(output_names)} metrics but returned {len(objective_values[0])}."
                )

        metrics_array = torch.tensor(objective_values)[:, : len(self.metric_names)]
        shift = 0
        if self.enable_feasible_predictor:
            feasible_array = torch.tensor(objective_values)[
                :, len(self.metric_names)
            ].to(dtype=torch.bool)
            shift += 1
        else:
            feasible_array = torch.Tensor()
        if self.settings["GP_model"] in ["botorch_fixed", "botorch_heteroskedastic"]:
            std_estimate_array = torch.tensor(objective_values)[
                :, len(self.metric_names) + shift
            ]
            shift += 1
        else:
            std_estimate_array = torch.Tensor()

        timestamp_array = torch.tensor(timestamps)
        data_array = DataArray(
            configurations,
            metrics_array,
            timestamp_array,
            feasible_array,
            std_estimate_array,
        )
        write_data_array(self, data_array, output_data_file)
        return data_array

    def print_data_array(self, data_array):
        """
        Print the data array.

        Input:
            - data_array: the data array to print. A dict of lists.
        """
        keys = ""
        for key in self.all_names:
            keys += str(key) + ","
        print(keys[:-1])
        configurations = self.convert(data_array.parameters_array, "internal", "string")
        for i in range(len(configurations)):
            print(
                *(
                    configurations[i]
                    + data_array.metrics_array[i].tolist()
                    + (
                        [data_array.feasible_array[i].item()]
                        if data_array.feasible_array.tolist()
                        else []
                    )
                    + [data_array.timestamp_array[i].item()]
                ),
                sep=",",
            )
        print()

    @staticmethod
    def current_milli_time():
        return int(round(time.time() * 1000))

    @staticmethod
    def get_unique_hash_string_from_values(configuration: torch.Tensor):
        """
        Returns a string that identifies a configuration in an unambiguous way.
        :param configuration: dictionary containing the pairs key/value.
        """
        return "_".join([str(x) for x in configuration.numpy()])


######################################
# LOAD AND SAVE
######################################


def write_data_array(param_space, data_array, filename):
    """
    Write a data array to a csv file.
    If a filename is not given, the output_data_file given in the json will be used.
    If the file does not exist, it will be created.

    Input:
        - data_array: the data array to write
        - filename: the file where data will be written
    """
    try:
        with open(filename, "a") as f:
            w = csv.writer(f)
            configurations = param_space.convert(
                data_array.parameters_array, "internal", "string"
            )
            for i in range(len(configurations)):
                w.writerow(
                    configurations[i]
                    + [m.item() for m in data_array.metrics_array[i]]
                    + (
                        [data_array.feasible_array[i].item()]
                        if param_space.enable_feasible_predictor
                        else []
                    )
                    + [data_array.timestamp_array[i].item()]
                    + (
                        [data_array.std_estimate[i].item()]
                        if param_space.settings["GP_model"]
                        in ["botorch_fixed", "botorch_heteroskedastic"]
                        else []
                    )
                )
    except Exception as e:
        print(e)
        raise Exception("Failed to write data array")
