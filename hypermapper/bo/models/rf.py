from typing import Dict, List, Any

import numpy as np
import torch

# makes it faster, but introduced some bugs
# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except:
#     pass

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from hypermapper.param.space import Space
from hypermapper.param.transformations import preprocess_parameters_array


class RFRegressionModel(RandomForestRegressor):
    """
    Implementation of our adapted RF model. We extend scikit-learns RF implementation to
    """

    def __init__(
        self,
        use_all_data_to_fit_mean=False,
        use_all_data_to_fit_variance=False,
        **kwargs
    ):
        """
        Input:
            use_all_data_to_fit_mean:
            use_all_data_to_fit_variance:
            **kwargs:
        """
        RandomForestRegressor.__init__(self, **kwargs)
        self.leaf_means = []
        self.leaf_variances = []
        self.use_all_data_to_fit_mean = use_all_data_to_fit_mean
        self.use_all_data_to_fit_variance = use_all_data_to_fit_variance

        # This is just to make the code faster. If the min_samples_split is 2, we can just set the internal mean to 0.
        if self.min_samples_split == 2 and not self.use_all_data_to_fit_mean:
            self.zero_internal_mean = True
        else:
            self.zero_internal_mean = False

    def update_leaf_values(
            self,
            X: np.array,
            y: np.array,
            use_all_data_to_fit_variance: bool = False,
    ):
        """
        The SKLearn RF implementation only uses the bootstrapped data to compute the leaf values. This method updates the values
        such that after the bootstrapped data is used to build the trees, all available data is used instead.

        Input:
            - X_leaves: matrix with dimensions (number_of_samples, number_of_trees). Stores the leaf each sample fell into for each tree.
            - y: values of each sample used to build the forest.
        """

        X_leaves = self.apply(X)
        for tree_idx, tree in enumerate(self.estimators_):
            node_values = [[] for _ in range(tree.tree_.node_count)]
            for leaf, y_value in zip(X_leaves[:, tree_idx], y):
                node_values[leaf].append(y_value)
            for i in range(tree.tree_.node_count):
                if tree.tree_.children_left[i] == -1:
                    tree.tree_.value[i] = np.mean(node_values[i])
                    if use_all_data_to_fit_variance:
                        tree.tree_.impurity[i] = np.var(node_values[i])

    @staticmethod
    def get_node_visits(tree: Any, x_leaves: np.ndarray):
        """
        Compute which samples passed through each node in a tree.

        Input:
            - tree: sklearn regression tree.
            - x_leaves: matrix with dimensions number_of_samples. Stores the leaf each sample fell into for each tree.

        Returns:
            - list of lists. Each internal list contains which samples went through the node represented by the index in the outer list.
        """
        node_count = tree.tree_.node_count
        node_visits = [[] for i in range(node_count)]
        for sample_idx in range(len(x_leaves)):
            leaf = x_leaves[sample_idx]
            node_visits[leaf].append(sample_idx)

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
                node_visits[parent] += node_visits[node_idx]
        return node_visits

    @staticmethod
    def get_node_bounds(samples: List[int], data_array: np.ndarray, threshold: float):
        """
        Compute the lower and upper bounds used to make a splitting decision at a tree node.

        Input:
            - samples: list containing the indices of all samples that went through the node.
            - data_array: list containing the values of one parameter for all the samples from the data.
            - threshold: original threshold used to split the node.

        Returns:
            - lower and upper bound that were used to compute the split.
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

    def fit_rf(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
    ):
        """
        Fit the adapted RF model.

        Input:
            - X: the training data for the RF model.
            - y: the training data labels for the RF model.
            - data_array: a dictionary containing previously explored points and their function values.
        """
        X, y = X.numpy(), y.numpy()
        self.estimator.splitter = "random"
        self.fit(X, y)
        if self.use_all_data_to_fit_mean:
            self.update_leaf_values(X, y, self.use_all_data_to_fit_variance)

    def get_mean_and_std(
        self,
        X: torch.Tensor,
        _,
        use_var=False
    ):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with an RF model.

        Input:
            - X: torch tensor array containing points to predict.
            - _: argument for GPs
            - var: whether to compute variance or standard deviation.

        Returns:
            - the predicted mean and uncertainty for each point.
        """

        # mean = self.predict(X)
        var = np.zeros(len(X))

        means = []
        for tree in self.estimators_:
            if self.zero_internal_mean:
                var_tree = np.zeros(X.shape[0])
            else:
                var_tree = tree.tree_.impurity[tree.apply(X)]
            var_tree[var_tree < 0] = 0
            mean_tree = tree.predict(X)
            var += var_tree + mean_tree ** 2
            means.append(mean_tree)
        mean = np.mean(np.array(means), axis=0)
        var /= self.n_estimators
        var -= mean ** 2.0
        var[var < 0.0] = 0.0

        if use_var:
            uncertainty = var
        else:
            uncertainty = np.sqrt(var)

        return torch.tensor(mean), torch.tensor(uncertainty)


class RFClassificationModel(RandomForestClassifier):

    def __init__(
            self,
            settings: Dict,
            param_space: Space,
            X: torch.Tensor,
            y: torch.Tensor,
    ):
        """
        Fit a Random Forest classification model.

        Input:
            - settings:
            - param_space: parameter space object for the current application.
            - X: input training data
            - y: boolean feasibility training values
        """
        self.param_space = param_space
        self.settings = settings
        self.X = X
        self.y = y
        n_estimators = 50
        max_features = 0.75

        class_weight = {True: 0.8, False: 0.2}
        RandomForestClassifier.__init__(
            self,
            class_weight=class_weight,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=False,
        )
        self.fit(self.X.numpy(), self.y.numpy())

    def feas_probability(self, data):
        """
        Compute the predictions of a model over a data array.

        Input:
            - data: data array with points to be predicted.

        Returns:
            - tensor with model predictions.
        """
        normalized_data, names = preprocess_parameters_array(data, self.param_space)
        return torch.tensor(self.predict_proba(normalized_data.numpy()))[:, list(self.classes_).index(True)]
