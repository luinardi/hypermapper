import copy
import datetime
import os
import sys
import numpy as np
import warnings
from collections import OrderedDict, defaultdict

import GPy
from scipy import stats

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

# ensure backward compatibility
try:
    from hypermapper.utility_functions import (
        data_dictionary_to_tuple,
        concatenate_list_of_dictionaries,
        domain_decomposition_and_parallel_computation,
        data_tuples_to_dictionary,
    )
    from hypermapper.local_search import local_search
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
        data_dictionary_to_tuple,
        concatenate_list_of_dictionaries,
        domain_decomposition_and_parallel_computation,
        data_tuples_to_dictionary,
    )
    from hypermapper.local_search import local_search


class RFModel(RandomForestRegressor):
    """
    Implementation of our adapted RF model. We extend scikit-learn's RF implementation to
    implement the adapted RF model proposed by Hutter et al.: https://arxiv.org/abs/1211.0906
    """

    def __init__(self, **kwargs):
        RandomForestRegressor.__init__(self, **kwargs)
        self.tree_means_per_leaf = []
        self.tree_vars_per_leaf = []

    def set_means_per_leaf(self, samples, leaf_per_sample):
        """
        Compute the mean value for each leaf in the forest.
        :param samples: list with the value of each sample used to build the forest.
        :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
        :return: list of number_of_trees dictionaries. Each dictionary contains the means for each leaf in a tree.
        """
        number_of_trees, number_of_samples = leaf_per_sample.shape
        for tree_idx in range(number_of_trees):
            leaf_means = defaultdict(int)
            leaf_sample_count = defaultdict(int)
            for sample_idx in range(number_of_samples):
                leaf = leaf_per_sample[tree_idx, sample_idx]
                leaf_sample_count[leaf] += 1
                leaf_means[leaf] += samples[sample_idx]
            for leaf in leaf_sample_count.keys():
                leaf_means[leaf] = leaf_means[leaf] / leaf_sample_count[leaf]
            self.tree_means_per_leaf.append(leaf_means)

    def get_means_per_leaf(self):
        return self.tree_means_per_leaf

    def set_vars_per_leaf(self, samples, leaf_per_sample):
        """
        Compute the variance for each leaf in the forest.
        :param samples: list with the value of each sample used to build the forest.
        :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
        :return: list of number_of_trees dictionaries. Each dictionary contains the variance for each leaf in a tree.
        """
        number_of_trees, number_of_samples = leaf_per_sample.shape
        for tree_idx in range(number_of_trees):
            samples_per_leaf = defaultdict(list)
            for sample_idx in range(number_of_samples):
                leaf = leaf_per_sample[tree_idx, sample_idx]
                samples_per_leaf[leaf].append(samples[sample_idx])

            leaf_vars = {}
            for leaf in samples_per_leaf.keys():
                if len(samples_per_leaf[leaf]) > 1:
                    leaf_vars[leaf] = np.var(samples_per_leaf[leaf], ddof=1)
                else:
                    leaf_vars[leaf] = 0
                # leaf_vars[leaf] = max(leaf_vars[leaf], 0.01) # This makes HyperMapper exploit too much. We will revisit this.
            self.tree_vars_per_leaf.append(leaf_vars)

    def get_vars_per_leaf(self):
        return self.tree_vars_per_leaf

    def get_samples_per_node(self, tree, leaf_per_sample):
        """
        Compute which samples passed through each node in a tree.
        :param tree: sklearn regression tree.
        :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
        :return: list of lists. Each internal list contains which samples went through the node represented by the index in the outer list.
        """
        node_count = tree.tree_.node_count
        samples_per_node = [[] for i in range(node_count)]
        for sample_idx in range(len(leaf_per_sample)):
            leaf = leaf_per_sample[sample_idx]
            samples_per_node[leaf].append(sample_idx)

        parents = [-1] * node_count
        left_children = tree.tree_.children_left
        right_children = tree.tree_.children_right
        for node_idx in range(node_count):
            if left_children[node_idx] != -1:
                parents[left_children[node_idx]] = node_idx
            if right_children[node_idx] != -1:
                parents[right_children[node_idx]] = node_idx

        for node_idx in range(node_count - 1, -1, -1):
            parent = parents[node_idx]
            if parent != -1:
                samples_per_node[parent] += samples_per_node[node_idx]
        return samples_per_node

    def get_node_bounds(self, samples, data_array, threshold):
        """
        Compute the lower and upper bounds used to make a splitting decision at a tree node.
        :param samples: list containing the indices of all samples that went through the node.
        :param data_array: list containing the values of one parameter for all of the samples from the data.
        :param threshold: original threshold used to split the node.
        :return: lower and upper bound that were used to compute the split.
        """
        lower_bound = float("-inf")
        upper_bound = float("inf")
        for sample in samples:
            sample_value = data_array[sample]
            if sample_value <= threshold:
                lower_bound = max(lower_bound, data_array[sample])
            else:
                upper_bound = min(upper_bound, data_array[sample])

        return lower_bound, upper_bound

    def get_leaves_per_sample(self, bufferx, param_space):
        """
        Compute in which leaf each sample falls into for each tree.
        :param bufferx: list containing the samples.
        :param param_space: parameter space object for the current application.
        :return: array containing the corresponding leaf of each tree for each sample.
        """
        leaf_per_sample = np.array([tree.apply(bufferx) for tree in self])
        return leaf_per_sample

    def fit_RF(self, X, y, data_array=None, param_space=None, **kwargs):
        """
        Fit the adapted RF model. If the data_array and param_space parameters are not provided
        a standard scikit-learn RF model will be fitted instead.
        :param X: the training data for the RF model.
        :param y: the training data labels for the RF model.
        :param data_array: a dictionary containing previously explored points and their function values.
        :param param_space: parameter space object for the current application.
        """
        self.fit(X, y, **kwargs)

        # If data_array and param_space are provided, fit the adapted RF
        if (data_array is not None) and (param_space is not None):
            bufferx = data_dictionary_to_tuple(data_array, list(data_array.keys()))
            leaf_per_sample = self.get_leaves_per_sample(bufferx, param_space)
            self.set_means_per_leaf(y, leaf_per_sample)
            self.set_vars_per_leaf(y, leaf_per_sample)
            new_features = list(data_array.keys())
            for tree_idx, tree in enumerate(self):
                samples_per_node = self.get_samples_per_node(
                    tree, leaf_per_sample[tree_idx, :]
                )

                left_children = tree.tree_.children_left
                right_children = tree.tree_.children_right
                for node_idx in range(tree.tree_.node_count):
                    if (
                        left_children[node_idx] == right_children[node_idx]
                    ):  # If both children are equal, this is a leaf in the tree
                        continue
                    feature = new_features[tree.tree_.feature[node_idx]]
                    threshold = tree.tree_.threshold[node_idx]

                    lower_bound, upper_bound = self.get_node_bounds(
                        samples_per_node[node_idx], data_array[feature], threshold
                    )
                    new_split = stats.uniform.rvs(
                        loc=lower_bound, scale=upper_bound - lower_bound
                    )
                    tree.tree_.threshold[node_idx] = new_split

    def compute_rf_prediction(self, leaf_per_sample):
        """
        Compute the forest prediction for a list of samples based on the mean of the leaves in each tree.
        :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
        :return: list containing the mean of each sample.
        """
        number_of_trees, number_of_samples = leaf_per_sample.shape
        sample_means = np.zeros(number_of_samples)
        tree_means_per_leaf = self.get_means_per_leaf()
        for tree_idx in range(number_of_trees):
            for sample_idx in range(number_of_samples):
                sample_leaf = leaf_per_sample[tree_idx, sample_idx]
                sample_means[sample_idx] += (
                    tree_means_per_leaf[tree_idx][sample_leaf] / number_of_trees
                )
        return sample_means

    def compute_rf_prediction_variance(self, leaf_per_sample, sample_means):
        """
        Compute the forest prediction variance for a list of samples based on the mean and variances of the leaves in each tree.
        The variance is computed as proposed by Hutter et al. in https://arxiv.org/pdf/1211.0906.pdf.
        :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
        :param sample_means: list containing the mean of each sample.
        :return: list containing the variance of each sample.
        """
        number_of_trees, number_of_samples = leaf_per_sample.shape
        mean_of_the_vars = np.zeros(number_of_samples)
        var_of_the_means = np.zeros(number_of_samples)
        sample_vars = np.zeros(number_of_samples)
        tree_means_per_leaf = self.get_means_per_leaf()
        tree_vars_per_leaf = self.get_vars_per_leaf()
        for sample_idx in range(number_of_samples):
            for tree_idx in range(number_of_trees):
                sample_leaf = leaf_per_sample[tree_idx, sample_idx]
                mean_of_the_vars[sample_idx] += (
                    tree_vars_per_leaf[tree_idx][sample_leaf] / number_of_trees
                )
                var_of_the_means[sample_idx] += (
                    tree_means_per_leaf[tree_idx][sample_leaf] ** 2
                ) / number_of_trees

            var_of_the_means[sample_idx] = abs(
                var_of_the_means[sample_idx] - sample_means[sample_idx] ** 2
            )
            sample_vars[sample_idx] = (
                mean_of_the_vars[sample_idx] + var_of_the_means[sample_idx]
            )
            if sample_vars[sample_idx] == 0:
                sample_vars[sample_idx] = 0.00001

        return sample_vars


def generate_multi_output_regression_model(
    data_array,
    param_space,
    Xcols,
    Ycols,
    learn_ratio,
    debug=False,
    n_estimators=10,
    max_features=0.5,
    customRegressor=RandomForestRegressor,
    print_importances=False,
):
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices).
    This method fits a single multi-output model for all objectives.
    :param data_array: the data to use for training.
    :param Xcols: the names of the input features used for training.
    :param Ycols: the names of the output labels used for training.
    :param learn_ratio: percentage of the input vectors used for training. A part of it maybe left over for cross validation.
    :param debug: is debugging mode enabled?
    :param n_estimators: number of trees.
    :param max_features: this is a parameter of the Random Forest. It decides how many feature to randomize.
    :param customRegressor: regression model to be used
    :return: 3 variables: the classifier, X_test , Y_test.
    """
    start_time = datetime.datetime.now()

    if param_space.get_input_normalization_flag() is True:
        compute_mean_and_std(data_array, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, Xcols)
    X = [preprocessed_data_array[param] for param in preprocessed_data_array]
    X = list(map(list, list(zip(*X))))
    Y = [data_array[Ycol] for Ycol in Ycols]
    Y = list(map(list, list(zip(*Y))))

    learn_size = int(len(X) * learn_ratio)
    X_train = X[0:learn_size]
    X_test = X[learn_size:]
    y_train = Y[0:learn_size]
    Y_test = Y[learn_size:]

    if len(X_test) == 0:
        X_test = X[:]
    if len(Y_test) == 0:
        Y_test = Y[:]

    regressor = customRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        n_jobs=1,
        bootstrap=False,
        min_samples_split=5,
    )
    regressor.fit(X_train, y_train)

    if print_importances:
        parameter_importances = compute_parameter_importance(
            regressor, Xcols, param_space
        )
        print(
            "Regression model on "
            + str(Ycols)
            + ". Features names: "
            + str(Xcols)
            + ", feature importances: "
            + str(parameter_importances)
        )
    sys.stdout.write_to_logfile(
        (
            "End of training - Time %10.2f sec\n"
            % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )

    return regressor, X_test, Y_test


def generate_mono_output_regression_models(
    data_array,
    param_space,
    Xcols,
    Ycols,
    learn_ratio,
    config,
    debug=False,
    model_type="random_forest",
    number_of_cpus=0,
    print_importances=False,
    normalize_objectives=False,
    objective_limits=None,
    **model_kwargs
):
    """
    Fit a regression model, supported model types are Random Forest and Gaussian Process.
    This method fits one mono output model for each objective.
    :param data_array: the data to use for training.
    :param Xcols: the names of the input features used for training.
    :param Ycols: the names of the output labels used for training.
    :param learn_ratio: percentage of the input vectors used for training. A part of it maybe left over for cross validation.
    :param debug: is debugging mode enabled?
    :param model_type: type of model to create. Either random_forest or gaussian_process.
    :param number_of_cpus: number of cpus to use in the models.
    :param print_importances: whether to print the importance of each parameter according to the model.
    :param normalize_objectives: whether to normalize the objective data before fitting the model.
    :param objective_limits: limits for normalizing the objective data.
    :return: 3 variables: the classifier, X_test , Y_test.
    """
    start_time = datetime.datetime.now()

    normalized_data_array = copy.deepcopy(data_array)
    if normalize_objectives:
        if objective_limits is not None:
            for col in Ycols:
                # Both limits are the same only if all elements in the array are equal. This causes the normalization to divide by 0.
                # We cannot optimize an objective when all values are the same, so we set it to 0
                if objective_limits[col][1] == objective_limits[col][0]:
                    normalized_col = [0] * len(normalized_data_array[col])
                else:
                    normalized_col = [
                        (x - objective_limits[col][0])
                        / (objective_limits[col][1] - objective_limits[col][0])
                        for x in normalized_data_array[col]
                    ]
                normalized_data_array[col] = normalized_col
        else:
            sys.stdout.write_to_logfile(
                "Warning: no limits provided, skipping objective normalization.\n"
            )

    if param_space.get_input_normalization_flag() is True:
        compute_mean_and_std(data_array, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, Xcols)
    regressor = {}
    X = [preprocessed_data_array[param] for param in preprocessed_data_array]
    X = list(map(list, list(zip(*X))))
    learn_size = int(len(X) * learn_ratio)
    X_train = X[0:learn_size]
    X_test = X[learn_size:]
    Y_test = {}

    if len(X_test) == 0:
        X_test = X[:]

    for i, Ycol in enumerate(Ycols):
        Y = normalized_data_array[Ycol]
        y_train = Y[0:learn_size]
        Y_test[Ycol] = Y[learn_size:]
        if len(Y_test[Ycol]) == 0:
            Y_test[Ycol] = Y[:]

        if debug:
            print(
                "Metric:%s, prepare training: len(X)=%s, len(X_train)=%s (learn_size=%s), len(X_test)=%s"
                % (Ycol, len(X), len(X_train), learn_size, len(X_test))
            )
            print(("Prepare training packages len(X) = %s" % len(X)))
            if i == 0:
                print("X_train")
                print(X_train)
            print("Y_train")
            print(y_train)
            print("Run accuracy prediction training...")

        if model_type == "gaussian_process":
            X_train = np.array(X_train)
            y_train = np.array(y_train).reshape(-1, 1)
            regressor[Ycol] = GPy.models.GPRegression(
                X_train,
                y_train,
                kernel=GPy.kern.Matern52(len(preprocessed_data_array), ARD=True),
                normalizer=True,
                **model_kwargs
            )
            if not config["noise"]:
                regressor[Ycol].Gaussian_noise.variance = 0
                regressor[Ycol].Gaussian_noise.fix()
            with np.errstate(
                divide="ignore", over="ignore", invalid="ignore"
            ):  # GPy's optimize has uncaught warnings that do not affect performance, suppress them so that they do not propagate to HyperMapper
                regressor[Ycol].optimize()
            if print_importances:
                print("Feature importance is currently not supported with GPs")
        elif model_type == "random_forest":
            n_estimators = config["models"]["number_of_trees"]
            max_features = config["models"]["max_features"]
            bootstrap = config["models"]["bootstrap"]
            min_samples_split = config["models"]["min_samples_split"]
            regressor[Ycol] = RFModel(
                n_estimators=n_estimators,
                max_features=max_features,
                bootstrap=bootstrap,
                min_samples_split=min_samples_split,
                **model_kwargs
            )
            regressor[Ycol].fit_RF(
                X_train,
                y_train,
                data_array=preprocessed_data_array,
                param_space=param_space,
            )
            if print_importances:
                parameter_importances = compute_parameter_importance(
                    regressor[Ycol], Xcols, param_space
                )
                print(
                    "Regression model on "
                    + str(Ycol)
                    + ". Features names: "
                    + str(Xcols)
                    + ", feature importances: "
                    + str(parameter_importances)
                )
        else:
            print("Unrecognized model type:", RandomForestRegressor)

    sys.stdout.write_to_logfile(
        (
            "End of training - Time %10.2f sec\n"
            % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )
    return regressor, X_test, Y_test


def generate_classification_model(
    application_name,
    param_space,
    data_array,
    Xcols,
    Ycols,
    learn_ratio,
    config,
    debug=False,
    n_estimators=15,
    max_features=0.5,
    customClassifier=ExtraTreesRegressor,
    number_of_cpus=0,
    data_array_exhaustive=None,
    enable_feasible_predictor_grid_search_on_recall_and_precision=False,
    feasible_predictor_grid_search_validation_file="",
    print_importances=False,
):
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices).
    :param application_name: the name of the application given by the json file.
    :param param_space: parameter space object for the current application.
    :param data_array: the data to use for training.
    :param Xcols: the names of the input features used for training.
    :param Ycols: the names of the output labels used for training.
    :param learn_ratio: percentage of the input vectors used for training. A part of it maybe left over for cross validation.
    :param debug: is debugging mode enabled?
    :param n_estimators: number of trees.
    :param max_features: this is a parameter of the Random Forest. It decides how many feature to randomize.
    :param customClassifier:
    :param number_of_cpus:
    :param enable_feasible_predictor_grid_search_on_recall_and_precision: does grid search on recall and precision to study the quality of the classifier.
    :param feasible_predictor_grid_search_validation_file: provides the dataset file name of the grid search cross-validation dataset.
    :return: 3 variables: the classifier, X_test , Y_test.
    """
    start_time = datetime.datetime.now()

    if param_space.get_input_normalization_flag() is True:
        compute_mean_and_std(data_array, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, Xcols)

    classifier_baggedtrees = {}
    X = [preprocessed_data_array[param] for param in preprocessed_data_array]
    X = list(map(list, list(zip(*X))))
    learn_size = int(len(X) * learn_ratio)
    X_train = X[0:learn_size]
    X_test = X[learn_size:]
    Y_test = {}

    if len(X_test) == 0:
        X_test = X[:]

    for i, Ycol in enumerate(Ycols):
        Y = data_array[Ycol]
        y_train = Y[0:learn_size]
        Y_test[Ycol] = Y[learn_size:]
        if len(Y_test[Ycol]) == 0:
            Y_test[Ycol] = Y[:]

        if debug:
            print(
                "Metric:%s, prepare training: len(X)=%s, len(X_train)=%s (learn_size=%s), len(X_test)=%s"
                % (Ycol, len(X), len(X_train), learn_size, len(X_test))
            )
            print(("Prepare training packages len(X) = %s" % len(X)))
            if i == 0:
                print("X_train")
                print(X_train)
            print("Y_train")
            print(y_train)
            print("Run accuracy prediction training...")

        class_weight = {True: 0.9, False: 0.1}
        classifier_baggedtrees[Ycol] = RandomForestClassifier(
            class_weight=class_weight, n_estimators=10, max_features=0.75
        )
        classifier_baggedtrees[Ycol].fit(X_train, y_train)

        if data_array_exhaustive != None:
            preprocessed_x_exhaustive = preprocess_data_array(
                data_array_exhaustive, param_space, Xcols
            )
            X_exhaustive = [
                preprocessed_x_exhaustive[param] for param in preprocessed_x_exhaustive
            ]
            X_exhaustive = list(map(list, list(zip(*X_exhaustive))))
            for i, Ycol in enumerate(Ycols):
                y_exhaustive = data_array_exhaustive[Ycol]
                print(
                    "Score of the feasibility classifier: "
                    + str(
                        classifier_baggedtrees[Ycol].score(X_exhaustive, y_exhaustive)
                    )
                )

        if print_importances:
            parameter_importances = compute_parameter_importance(
                classifier_baggedtrees[Ycol], Xcols, param_space
            )
            print(
                "Classification model. Features names: "
                + str(Xcols)
                + ", feature importances: "
                + str(parameter_importances)
            )

        if enable_feasible_predictor_grid_search_on_recall_and_precision:
            dataset = feasible_predictor_grid_search_validation_file
            compute_recall_and_precision_on_RF_hyperparameters(
                dataset, param_space, X_train, y_train
            )

    sys.stdout.write_to_logfile(
        (
            "End of training - Time %10.2f sec\n"
            % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )
    return classifier_baggedtrees, X_test, Y_test


def compute_recall_and_precision_on_RF_hyperparameters(
    dataset, param_space, X_train, y_train
):
    """
    Compute recall and precision for the binary random forests classifier using cross validation.
    Reference: https://en.wikipedia.org/wiki/Precision_and_recall
    This function should be used only for debugging and development purposes.
    For debugging to see if a new application performs well on the set of hyperparameters of the random forests binary classifier.
    For development to set these hyperparameters.
    The objective in HyperMapper is to maximize the recall because we don't want to loose the opportunity of finding good samples.
    At the same time we should keep under control the precision for efficiency (we don't want too many false positives
    because in this case the effect would be like not having the classifier at all).
    Notice that using the accuracy instead of the recall would be the wrong thing to do because we care more about the
    fact that we want to classify precisely the true class (we care less of the false class).
    :param dataset: the test data to use to check the recall and precision.
    :param param_space: parameter space object for the current application.
    :param X_train: the features data used to train the Random Forests.
    :param y_train: the labels data used to train the Random Forests.
    :return:
    """
    start_time = datetime.datetime.now()
    learn_ratio = 0.75
    Xcols = param_space.get_input_parameters()
    Ycol = param_space.get_feasible_parameter()[0]
    print("#######################################################################")
    print("####### Start of the cross-validation for the RF classifier ###########")
    print(("Loading data from %s ..." % dataset))
    data_array, fast_addressing_of_data_array = param_space.load_data_file(dataset)
    count = 0
    for i in data_array[Ycol]:
        if i == True:
            count += 1
    print("\nCount of feasible in the dataset file %s = %d" % (dataset, count))
    # Set the parameters by cross-validation
    tuned_parameters = [
        {
            "n_estimators": [10, 100, 1000],
            "max_features": ["auto", 0.5, 0.75],
            "max_depth": [None, 4, 8],
            "class_weight": [
                {True: 0.50, False: 0.50},
                {True: 0.75, False: 0.25},
                {True: 0.9, False: 0.1},
            ],
        }
    ]
    X_test = [data_array[Xcol] for Xcol in Xcols]
    X_test = list(map(list, list(zip(*X_test))))
    Y_test = data_array[Ycol]
    scores = ["recall", "precision"]
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(
            RandomForestClassifier(),
            tuned_parameters,
            cv=5,
            n_jobs=-1,
            scoring="%s_macro" % score,
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found:")
        print(clf.best_params_)

        print("Grid scores:")
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print("Detailed classification report:")
        print(
            "The model is trained on the HyperMapper samples, total samples: %d"
            % len(X_train)
        )
        print("The scores are computed on the test set.")
        y_true, y_pred = Y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

    end_time = datetime.datetime.now()
    print(
        (
            "\nTotal time to compute the cross-validation for the Random Forests binary classifier is: "
            + str((end_time - start_time).total_seconds())
            + " seconds"
        )
    )
    print("####### End of the cross-validation for the RF classifier ###########")
    print("#######################################################################")


def compute_parameter_importance(model, input_params, param_space):
    """
    Compute the importance for each input parameter in a RF model.
    For now, importance can only be computed for RF models.
    :param model: a RF model.
    :param input_params: the input parameters whose importance we want to compute
    :param param_space: parameter space object for the current application.
    :return: list containing the importance of each input parameter.
    """
    parameter_importances = [0] * len(input_params)
    categorical_parameters = param_space.get_input_categorical_parameters_objects(
        input_params
    )
    feature_idx = 0
    for idx, param in enumerate(input_params):
        if param in categorical_parameters:
            size = categorical_parameters[param].get_size()
            parameter_importances[idx] = sum(
                model.feature_importances_[feature_idx : feature_idx + size]
            )
            feature_idx = feature_idx + size
        else:
            parameter_importances[idx] = model.feature_importances_[feature_idx]
            feature_idx += 1
    return parameter_importances


def compute_model_mean_and_uncertainty(
    bufferx, model, model_type, param_space, var=False
):
    """
    Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points.
    :param bufferx: list containing points to predict.
    :param model: model to use for the prediction.
    :param model_type: the type of the model being used. For now, either random_forest or gaussian_process.
    :param param_space: parameter space object for the current application.
    :param var: whether to compute variance or standard deviation.
    :return: the predicted mean and uncertaitny for each point.
    """
    if model_type == "random_forest":
        mean, uncertainty = compute_rf_prediction_mean_and_uncertainty(
            bufferx, model, param_space, var=var
        )
    if model_type == "gaussian_process":
        mean, uncertainty = compute_gp_prediction_mean_and_uncertainty(
            bufferx, model, param_space, var=var
        )
    for objective in mean:
        mean[objective][np.isnan(mean[objective])] = sys.maxsize
    return mean, uncertainty


def compute_rf_prediction_mean_and_uncertainty(bufferx, model, param_space, var=False):
    """
    Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a RF model.
    :param bufferx: list containing points to predict.
    :param model: model to use for the prediction.
    :param param_space: parameter space object for the current application.
    :param var: whether to compute variance or standard deviation.
    :return: the predicted mean and uncertaitny for each point.
    """
    prediction_means = {}
    prediction_uncertainty = {}
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    for objective in model:
        leaf_per_sample = model[objective].get_leaves_per_sample(
            normalized_bufferx, param_space
        )
        prediction_means[objective] = model[objective].compute_rf_prediction(
            leaf_per_sample
        )
        prediction_variances = model[objective].compute_rf_prediction_variance(
            leaf_per_sample, prediction_means[objective]
        )
        if var:
            prediction_uncertainty[objective] = prediction_variances
        else:
            prediction_uncertainty[objective] = np.sqrt(prediction_variances)

    return prediction_means, prediction_uncertainty


def sample_model_posterior(bufferx, model, model_type, param_space):
    """
    Compute the value of a number of points according to a function sapmled from the model posterior.
    :param bufferx: list containing points to predict.
    :param model: model to use for the prediction.
    :param model_type: the type of the model being used. For now, either random_forest or gaussian_process.
    :param param_space: parameter space object for the current application.
    :return: the predicted value of each point.
    """
    if model_type == "random_forest":
        model_predictions = model_prediction(bufferx, model, param_space)
    elif model_type == "gaussian_process":
        model_predictions = sample_gp_posterior(bufferx, model, param_space)
    return model_predictions


def parallel_model_prediction(
    model, bufferx, param_space, debug=False, number_of_cpus=0
):
    """
    This function explicitly parallelize the prediction of the Random Forest model.
    The parallelization is done by hand however not exploiting the fact that the RF model has an option n_jobs that automatically parallize the fit and prediction.
    That option seems not to work though.
    :param model: model (or dictionary of models) to use for prediction.
    :param bufferx: data to perform prediction on.
    :param optimization_metrics: the names of the objectives being optimized.
    :param debug: turn debug mode on/off.
    :param number_of_cpus: number of cpus to use in parallel.
    :return:
    """
    if type(model) is dict:
        return domain_decomposition_and_parallel_computation(
            debug,
            mono_output_model_prediction,
            concatenate_function_prediction,
            bufferx,
            number_of_cpus,
            model,
            param_space,
        )
    else:
        return domain_decomposition_and_parallel_computation(
            debug,
            multi_output_model_prediction,
            concatenate_function_prediction,
            bufferx,
            number_of_cpus,
            model,
            param_space,
        )


def multi_output_model_prediction(bufferx, model, param_space):
    """
    :param bufferx: list containing points to predict.
    :param model: model to use for prediction.
    :param optimization_metrics: list containing the metrics being optimized.
    :return: dictionary containing predictions for each objective.
    """
    optimization_metrics = param_space.get_optimization_parameters()
    input_params = param_space.get_input_parameters()
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    predictions = model.predict(normalized_bufferx)
    for idx, parameter in enumerate(optimization_metrics):
        Cresult[parameter] = predictions[:, idx]
    return Cresult


def mono_output_model_prediction(bufferx, model, param_space):
    """
    :param bufferx: list containing points to predict.
    :param model: model to use for prediction.
    :return: dictionary containing predictions for each objective.
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    for parameter in model:
        Cresult[parameter] = model[parameter].predict(normalized_bufferx)
    return Cresult


def model_prediction(bufferx, model, param_space):
    """
    Compute the predictions of a model over a data array.
    :param bufferx: data array with points to be predicted.
    :param model: model to use to perform prediction.
    :return: array with model predictions.
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    for parameter in model:
        Cresult[parameter] = model[parameter].predict(normalized_bufferx)
    return Cresult


def model_probabilities(bufferx, model, param_space):
    """
    Compute the predictions of a model over a data array.
    :param bufferx: data array with points to be predicted.
    :param model: model to use to perform prediction.
    :return: array with model predictions.
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    for parameter in model:
        Cresult[parameter] = model[parameter].predict_proba(normalized_bufferx)
    return Cresult


def concatenate_function_prediction(results_parallel):
    """
    Concatenate the results of parallel predictions into a single data dictionary.
    :param bufferx: data array with points to be predicted.
    :return: dictionary containing concatenated results.
    """
    concatenate_result = {}
    for key in list(results_parallel[0].keys()):
        concatenate_result[key] = np.concatenate(
            [results_parallel[chunk][key] for chunk in range(len(results_parallel))]
        )
    return concatenate_result


def preprocess_data_buffer(bufferx, param_space):
    """
    Preprocess an input buffer before feeding into a regression/classification model.
    The preprocessing standardize non-categorical inputs (if the flag is set).
    It also transforms categorical variables using one-hot encoding.
    :param bufferx: data array containing the input configurations to preprocess.
    :param param_space: parameter space object for the current application.
    :return: preprocessed data buffer.
    """
    input_params = param_space.get_input_parameters()
    data_array = data_tuples_to_dictionary(bufferx, input_params)
    preprocessed_data_array = preprocess_data_array(
        data_array, param_space, input_params
    )
    preprocessed_buffer = data_dictionary_to_tuple(
        preprocessed_data_array, list(preprocessed_data_array.keys())
    )
    return preprocessed_buffer


def preprocess_data_array(data_array, param_space, input_params):
    """
    Preprocess a data_array before feeding into a regression/classification model.
    The preprocessing standardize non-categorical inputs (if the flag is set).
    It also transforms categorical variables using one-hot encoding.
    :param data_array: dictionary containing the input configurations to preprocess.
    :param param_space: parameter space object for the current application.
    :param input_params: list with the names of the input parameters to preprocess in the data array.
    :return: preprocessed data array. The returned data array will contain only the keys in input_params.
    """
    non_categorical_parameters = param_space.get_input_non_categorical_parameters(
        input_params
    )
    parameter_objects = param_space.get_input_parameters_objects()
    categorical_parameters = param_space.get_input_categorical_parameters(input_params)
    preprocessed_data_array = {}
    for param in non_categorical_parameters:
        if param_space.get_type(param) == "ordinal":
            param_value_list = parameter_objects[param].get_values()
            param_size = parameter_objects[param].get_size()
            preprocessed_data_array[param] = []
            for param_value in data_array[param]:
                preprocessed_data_array[param].append(
                    param_value_list.index(param_value) / param_size
                )
        elif param_space.get_input_normalization_flag() is True:
            X = np.array(data_array[param], dtype=np.float64)
            mean = param_space.get_parameter_mean(param)
            std = param_space.get_parameter_std(param)
            X = (X - mean) / std
            preprocessed_data_array[param] = X
        else:
            preprocessed_data_array[param] = data_array[param]
    for param in categorical_parameters:
        # Categorical variables are encoded as their index, generate a list of "index labels"
        categories = np.arange(parameter_objects[param].get_size())
        encoder = OneHotEncoder(categories="auto", sparse=False)
        encoder.fit(categories.reshape(-1, 1))
        x = np.array(data_array[param]).reshape(-1, 1)
        encoded_x = encoder.transform(x)
        for i in range(encoded_x.shape[1]):
            new_key = param + "_" + str(categories[i])
            preprocessed_data_array[new_key] = list(encoded_x[:, i])
    return preprocessed_data_array


def compute_mean_and_std(data_array, param_space):
    input_params = param_space.get_input_parameters()
    for param in input_params:
        X = np.array(data_array[param], dtype=np.float64)
        mean = np.mean(X)
        std = np.std(X)
        param_space.set_parameter_mean(param, mean)
        param_space.set_parameter_std(param, std)


def compute_gp_prediction_mean_and_uncertainty(bufferx, model, param_space, var=False):
    """
    Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.
    :param bufferx: list containing points to predict.
    :param model: model to use for the prediction.
    :param param_space: parameter space object for the current application.
    :param var: whether to compute variance or standard deviation.
    :return: the predicted mean and uncertainty for each point
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    normalized_bufferx = np.array(normalized_bufferx)
    means = {}
    vars = {}
    uncertainty = {}
    for parameter in model:
        means[parameter], vars[parameter] = model[parameter].predict(normalized_bufferx)
        means[parameter] = means[parameter].flatten()
        vars[parameter] = vars[parameter].flatten()

        # Precision can sometimes lead GPy to predict extremely low deviation, which leads to numerical issues
        # We add a floor to std to avoid these numerical issues. The majority of std values observed are naturally above this floor.
        vars[parameter][vars[parameter] < 10**-11] = 10**-11
        if var:
            uncertainty[parameter] = vars[parameter]
        else:
            uncertainty[parameter] = np.sqrt(vars[parameter])

    return means, uncertainty


def sample_gp_posterior(bufferx, model, param_space):
    """
    Sample from the gp posterior for a list of points.
    :param bufferx: list containing points to predict.
    :param model: the GP regression model.
    :param param_space: parameter space object for the current application.
    :return: GP sample
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    normalized_bufferx = np.array(normalized_bufferx)
    gp_samples = {}
    for objective in model:
        gp_samples[objective] = model[objective].posterior_samples_f(
            normalized_bufferx, size=1
        )

    return gp_samples


def ls_compute_posterior_mean(configurations, model, model_type, param_space):
    """
    Compute the posterior mean for a list of configurations. This function follows the interface defined by
    HyperMapper's local search. It receives configurations from the local search and returns their values.
    :param configurations: configurations to compute posterior mean
    :param model: posterior model to use for predictions
    :param model_type: string with the type of model being used.
    :param param_space: Space object containing the search space.
    :return: the posterior mean value for each configuration. To satisfy the local search's requirements, also returns a list of feasibility flags, all set to 1.
    """
    configurations = concatenate_list_of_dictionaries(configurations)
    configurations = data_dictionary_to_tuple(
        configurations, param_space.get_input_parameters()
    )
    posterior_means, _ = compute_model_mean_and_uncertainty(
        configurations, model, model_type, param_space
    )

    objective = param_space.get_optimization_parameters()[0]
    return list(posterior_means[objective]), [1] * len(posterior_means[objective])


def minimize_posterior_mean(
    model,
    config,
    param_space,
    data_array,
    objective_limits,
    normalize_objectives,
    profiling,
):
    """
    Compute the minimum of the posterior model using a multi-start local search.
    :param model: posterior model to use for predictions
    :param config: the application scenario defined in the json file
    :param param_space: Space object containing the search space.
    :param data_array: array containing all of the points that have been explored
    :param objective_limits: estimated limits for the optimization objective, used to restore predictions to original range.
    :param normalize_objectives: whether objective values were normalized before fitting the model.
    :param profiling: whether to profile the local search run.
    :return: the best configuration according to the mean of the posterior model.
    """
    local_search_starting_points = config["local_search_starting_points"]
    local_search_random_points = config["local_search_random_points"]
    fast_addressing_of_data_array = (
        {}
    )  # We don't mind exploring repeated points in this case
    scalarization_key = config["scalarization_key"]
    number_of_cpus = config["number_of_cpus"]
    model_type = config["models"]["model"]

    optimization_function_parameters = {}
    optimization_function_parameters["model"] = model
    optimization_function_parameters["model_type"] = model_type
    optimization_function_parameters["param_space"] = param_space

    _, best_configuration = local_search(
        local_search_starting_points,
        local_search_random_points,
        param_space,
        fast_addressing_of_data_array,
        False,  # we do not want the local search to consider feasibility constraints
        ls_compute_posterior_mean,
        optimization_function_parameters,
        scalarization_key,
        number_of_cpus,
        previous_points=data_array,
        profiling=profiling,
    )

    objective = param_space.get_optimization_parameters()[0]
    best_configuration[objective] = ls_compute_posterior_mean(
        [best_configuration], model, model_type, param_space
    )[0][0]
    if normalize_objectives:
        objective_min, objective_max = (
            objective_limits[objective][0],
            objective_limits[objective][1],
        )
        best_configuration[objective] = (
            best_configuration[objective] * (objective_max - objective_min)
            + objective_min
        )

    return best_configuration
