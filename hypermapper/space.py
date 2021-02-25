import csv
import datetime
import itertools
import math
import os
import random
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from numbers import Number
from random import shuffle

import pandas as pd
import numpy as np
from numpy.random import beta
from scipy import stats
from scipy.stats import gaussian_kde, norm

# ensure backward compatibility
try:
    from hypermapper.utility_functions import (
        deal_with_relative_and_absolute_path,
        concatenate_data_dictionaries,
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
        deal_with_relative_and_absolute_path,
        concatenate_data_dictionaries,
    )


################# PARAMETERS CLASSES ##################
# The input search space supports four types of parameters: Categorical, Ordinal, Integer and Real.
class Parameter:
    """
    Super class of the different parameter classes.
    """

    def __init__(self):
        # Define the Beta distribution alpha and beta parameters:
        self.densities_alphas = {
            "uniform": 1.0,
            "gaussian": 3.0,
            "decay": 0.8,
            "exponential": 1,
        }
        self.densities_betas = {
            "uniform": 1.0,
            "gaussian": 3.0,
            "decay": 1,
            "exponential": 0.8,
        }


class RealParameter(Parameter):
    """
    This class defines a real (continuous) parameter.
    """

    def __init__(
        self,
        min_value,
        max_value,
        preferred_discretization,
        default,
        probability_distribution,
    ):
        """
        Initilization method. The possible values for this parameter are between min_value and max_value.
        :param min_value: minimum value.
        :param max_value: maximum value.
        :param preferred_discretization: list of discrete values.
        :param default: default value.
        :param probability_distribution: a string describing the probability density function.
        """
        Parameter.__init__(self)
        self.min_value = min_value
        self.max_value = max_value
        self.default = default
        self.preferred_discretization = preferred_discretization
        self.prior = probability_distribution
        self.estimated_pdf = None
        self.custom_gaussian_prior_mean = None
        self.custom_gaussian_prior_std = None
        self.alpha = None
        self.beta = None
        if isinstance(self.prior, str) and (self.prior in self.densities_alphas.keys()):
            self.alpha = self.densities_alphas[probability_distribution]
            self.beta = self.densities_betas[probability_distribution]

        self.cdf_distribution = None
        if isinstance(self.prior, list):
            self.prior = list(self.prior / np.sum(self.prior))
            self.param_xs = np.linspace(self.get_min(), self.get_max(), 10000)
            self.pdf_distribution = self.interpolate_pdf(self.param_xs)
            self.cdf_distribution = np.cumsum(self.pdf_distribution)
            self.cdf_distribution = self.cdf_distribution / self.cdf_distribution[-1]

    def randomly_select(self, size=1):
        """
        Sample from the prior distribution for this parameter. Prior sampling is performed by either estimating the distribution
        via KDE, specified by a Gaussian entered by the user or by the pre-determined beta distributions. For estimate or
        custom_gaussian, the samples are checked to be within the parameter bounds, and removed if they are not. If we end up
        with too few points after one batch of samples, another batch is sampled and added to the old one.
        :return: the random sampled values from the set of available values.
        :param size: the number of sampled random points
        """
        oversampling_factor = 5
        samples = np.array([])
        if isinstance(self.prior, list):
            x_probability = random.uniform(0, 1)
            return np.interp(x_probability, self.cdf_distribution, self.param_xs)
        elif self.prior == "estimate":
            while len(samples) < size:
                new_samples = self.estimated_pdf.resample(
                    size=size * oversampling_factor
                )[
                    0
                ]  # sampling too many points to ensure we will likely get many enough (and sampling more takes no time)
                inside_range = [
                    (new_samples <= self.max_value) & (new_samples >= self.min_value)
                ]
                samples = np.append(samples, new_samples[tuple(inside_range)])
            samples = samples[0:size]
        elif self.prior == "custom_gaussian":
            attempt_counter = 0
            while len(samples) < size:
                attempt_counter += 1
                if attempt_counter > 100:
                    raise SystemExit(
                        "Unable to sample points within range for the custom Gaussian - ensure the imput parameters are reasonable. Exiting."
                    )
                new_samples = np.array(
                    norm.rvs(
                        loc=self.custom_gaussian_prior_mean,
                        scale=self.custom_gaussian_prior_std,
                        size=size * oversampling_factor,
                    )
                )
                inside_range = [
                    (new_samples <= self.max_value) & (new_samples >= self.min_value)
                ]
                samples = np.append(samples, new_samples[tuple(inside_range)])

            # remove excess samples to get the proper amount
            samples = samples[0:size]
        else:
            samples = beta(self.alpha, self.beta, size)
            samples = self.from_range_0_1_to_parameter_value(samples)
        return samples

    def randomly_select_uniform(self, size=1):
        """
        Sample from an uniform distribution for this parameter.
        :param size: the number of sampled random points
        :return: the random sampled values from the set of available values.
        """
        samples = beta(1, 1, size)
        samples = self.from_range_0_1_to_parameter_value(samples)
        return samples

    def estimate_pdf(self, values, bw_method=None):
        """
        Estimate a kde prior for this parameter.
        :param values: the points to use to estimate the prior.
        :param bw_method: the bandwidth of the kde.
        """
        self.estimated_pdf = gaussian_kde(values, bw_method=bw_method)

    def interpolate_pdf(self, x):
        """
        Find the pdf for x by interpolating the list of pdf values in the prior.
        :param x: points to compute pdf.
        :return: interpolated pdf for x.
        """
        prior_xs = np.linspace(self.get_min(), self.get_max(), len(self.prior))
        return np.interp(x, prior_xs, self.prior)

    def pdf(self, x):
        """
        Compute the probability density of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        if isinstance(self.prior, list):
            return self.interpolate_pdf(x)
        elif self.prior == "estimate":
            return self.estimated_pdf(x)
        elif self.prior == "custom_gaussian":
            return norm.pdf(
                x,
                loc=self.custom_gaussian_prior_mean,
                scale=self.custom_gaussian_prior_std,
            )
        else:
            rescaled_x = self.from_parameter_value_to_0_1_range(x)
            return stats.beta.pdf(rescaled_x, self.alpha, self.beta)

    def cdf(self, x):
        t0_cdf = datetime.datetime.now()
        if self.prior == "estimate":
            return self.estimated_pdf.integrate_box_1d(float("-inf"), x)
        elif self.prior == "custom_gaussian":
            return norm.cdf(
                x,
                loc=self.custom_gaussian_prior_mean,
                scale=self.custom_gaussian_prior_std,
            )
        else:
            rescaled_x = self.from_parameter_value_to_0_1_range(x)
            return stats.beta.cdf(rescaled_x, self.alpha, self.beta)

    def get_x_probability(self, x):
        """
        Compute the probability of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        if isinstance(self.prior, list):
            return self.interpolate_pdf(x)
        elif self.prior == "estimate":
            return self.estimated_pdf(x)[0]
        elif self.prior == "custom_gaussian":
            return norm.pdf(
                x,
                loc=self.custom_gaussian_prior_mean,
                scale=self.custom_gaussian_prior_std,
            )
        else:
            rescaled_x = self.from_parameter_value_to_0_1_range(x)
            pdf = stats.beta.pdf(rescaled_x, self.alpha, self.beta)
            # when using the exponential or decay beta, values too close to 1 have pdf(x) = infinity, which leads to numerical issues
            # the highest value that can be reached before infinity for the exponential is ~1242, we cap the pdf at that.
            if pdf == float("inf"):
                pdf = 1242
            return pdf

    def get_size(self):
        return float("inf")

    def get_discrete_size(self):
        return len(self.preferred_discretization)

    def get_discrete_values(self):
        return self.preferred_discretization

    def get_default(self):
        return self.default

    def get_min(self):
        return self.min_value

    def get_max(self):
        return self.max_value

    def set_custom_gaussian_prior_params(self, mean, std):
        if std == -1:
            std = (self.max_value - self.min_value) / 2
        self.custom_gaussian_prior_mean = mean
        self.custom_gaussian_prior_std = std

    def from_range_0_1_to_parameter_value(self, X):
        """
        Scaling the values in X from ranges of [0, 1] to ranges defined by this parameter.
        :param X: a vector of of [0, 1] values.
        :return: the scaled values to the range of this parameter with respect to min and max.
        """
        if type(X) != np.ndarray and type(X) != list:
            X_copy = np.array([X]).copy()
        else:
            X_copy = X.copy()

        samples = X_copy * (self.get_max() - self.get_min()) + self.get_min()
        if len(X_copy) == 1:
            return samples[0]
        else:
            return samples

    def from_parameter_value_to_0_1_range(self, X):
        """
        Scaling the values in X from ranges of [0, 1] to ranges defined by this parameter.
        :param X: a vector of of [0, 1] values.
        :return: the scaled values to the range of this parameter with respect to min and max.
        """
        if type(X) == list:
            return (X[:] - self.get_min()) / (self.get_max() - self.get_min())
        else:
            return (X - self.get_min()) / (self.get_max() - self.get_min())


class IntegerParameter(Parameter):
    """
    This class defines a Integer parameter, i.e. an interval of integers from a to b.
    """

    def __init__(self, min_value, max_value, default, probability_distribution):
        """
        Initilization method. The possible values for this parameter are between min_value and max_value.
        :param min_value: minimum value.
        :param max_value: maximum value.
        :param default: default value.
        :param probability_distribution: a string describing the probability density function.
        """
        Parameter.__init__(self)
        self.min_value = min_value
        self.max_value = max_value
        self.default = default
        self.distribution = None
        self.values_list = list(range(min_value, max_value + 1))
        if isinstance(probability_distribution, str):
            self.prior = probability_distribution
            self.alpha = self.densities_alphas[probability_distribution]
            self.beta = self.densities_betas[probability_distribution]
        else:
            self.prior = "distribution"
            self.distribution = probability_distribution

    def randomly_select(self, size=1):
        """
        Sample from the specific beta distribution defined in the json for this parameter.
        :param size: the number of sampled random points
        :return: the random sampled values from the set of available values.
        """
        prior = self.prior
        if prior == "distribution":
            return np.random.choice(
                self.values_list, size=size, p=self.distribution
            ).astype(int)
        else:
            samples = beta(
                self.densities_alphas[prior], self.densities_betas[prior], size
            )
            return self.from_range_0_1_to_parameter_value(samples)

    def randomly_select_uniform(self, size=1):
        """
        Sample from the prior distribution for this parameter.
        :param size: the number of sampled random points
        :return: the random sampled values from the set of available values.
        """
        return np.random.choice(self.values_list, size=size).astype(int)

    def pdf(self, x):
        """
        Compute the probability density of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        prior = self.prior
        if prior == "distribution":
            values = self.values_list
            x_index = values.index(x)
            distribution = self.distribution
            return distribution[x_index]
        else:
            rescaled_x = self.from_parameter_value_to_0_1_range(x)
            return stats.beta.pdf(rescaled_x, self.alpha, self.beta)

    def get_x_probability(self, x):
        """
        Compute the probability of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        return self.pdf(x)

    def get_size(self):
        return self.max_value - self.min_value + 1

    def get_discrete_size(self):
        return self.get_size()

    def get_discrete_values(self):
        return range(self.min_value, self.max_value + 1)

    def get_default(self):
        return self.default

    def get_min(self):
        return self.min_value

    def get_max(self):
        return self.max_value

    def from_range_0_1_to_parameter_value(self, X):
        """
        Scaling the values in X from ranges of [0, 1] to ranges defined by this parameter.
        :param X: a numpy array of [0, 1] values.
        :return: the scaled values to the range of this parameter.
        """
        if type(X) != np.ndarray and type(X) != list:
            X_copy = np.array([X]).copy()
        else:
            X_copy = X.copy()

        samples = np.round(X_copy * self.get_size() - 0.5).astype(int)

        if len(samples) == 1:
            return samples[0]

        return samples

    def from_parameter_value_to_0_1_range(self, X):
        """
        Scaling the values in X from its original range to [0, 1].
        :param X: a vector of of parameter values.
        :return: the scaled values to the [0, 1] range.
        """
        if type(X) == list:
            return (X[:] - self.get_min()) / (self.get_max() - self.get_min())
        else:
            return (X - self.get_min()) / (self.get_max() - self.get_min())


class OrdinalParameter(Parameter):
    """
    This class defines an ordinal parameter, i.e. parameters that are numerical and can be ordered using lessser, equal, greater than.
    """

    def __init__(self, values, default, probability_distribution):
        """
        Initilization method. The possible values for this parameter are defined by the list values.
        :param values: list of possible value for this parameter.
        :param default: default value.
        :param probability_distribution: a string describing the probability density function or a list of values describing the probability distribution.
        """
        Parameter.__init__(self)
        self.values = sorted(values, key=float)  # ascending order
        self.default = default
        self.distribution = []
        if isinstance(probability_distribution, str):
            self.prior = probability_distribution
        else:
            self.prior = "distribution"
            self.distribution = probability_distribution

    def randomly_select(self, size=1):
        """
        Sample from the specific beta distribution defined in the json for this parameter.
        :param size: the number of sampled random points
        :return: the random sampled values from the set of available values.
        """
        prior = self.prior

        if prior == "distribution":
            return np.random.choice(self.values, size=size, p=self.distribution)
        else:
            samples = beta(
                self.densities_alphas[prior], self.densities_betas[prior], size
            )
            return self.from_range_0_1_to_parameter_value(samples)

    def randomly_select_uniform(self, size=1):
        """
        Sample from the prior distribution for this parameter.
        :param size: the number of sampled random points
        :return: the random sampled values from the set of available values.
        """
        return np.random.choice(self.values, size=size)

    def pdf(self, x):
        """
        Compute the probability of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        prior = self.prior
        values = self.get_values()
        x_index = values.index(x)
        if prior == "distribution":
            distribution = self.get_parameter_distribution()
            return distribution[x_index]
        else:
            lower_bound = x_index / len(values)
            upper_bound = (x_index + 1) / len(values)
            lower_cdf = stats.beta.cdf(
                lower_bound, self.densities_alphas[prior], self.densities_betas[prior]
            )
            upper_cdf = stats.beta.cdf(
                upper_bound, self.densities_alphas[prior], self.densities_betas[prior]
            )
            return upper_cdf - lower_cdf

    def get_x_probability(self, x):
        """
        Compute the probability of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        # For categorical parameters, the pdf gives a probability and not a density.
        return self.pdf(x)

    def get_size(self):
        return len(self.values)

    def get_discrete_size(self):
        return self.get_size()

    def get_discrete_values(self):
        return self.values

    def get_values(self):
        return self.values

    def get_default(self):
        return self.default

    def get_min(self):
        if self.get_size() > 0:
            return self.values[0]
        else:
            print("Error: this ordinal parameter doesn't have values. Exit.")
            exit()

    def get_max(self):
        if self.get_size() > 0:
            return self.values[self.get_size() - 1]
        else:
            print("Error: this ordinal parameter doesn't have values. Exit.")
            exit()

    def get_prior(self):
        return self.prior

    def get_parameter_distribution(self):
        return self.distribution

    def from_range_0_1_to_parameter_value(self, X):
        """
        Scaling the values in X from ranges of [0, 1] to ranges defined by this parameter.
        :param X: a numpy array of [0, 1] values.
        :return: the scaled values to the range of this parameter.
        """
        # get samples from a range - get them to integers and return the integer index

        if type(X) != np.ndarray and type(X) != list:
            X_copy = np.array([X]).copy()
        else:
            X_copy = X.copy()

        samples = np.round(X_copy * self.get_size() - 0.5).astype(int)
        if len(samples) == 1:
            return self.values[samples[0]]

        return np.array(self.values)[(samples)]


class CategoricalParameter(Parameter):
    """
    This class defines a categorical parameter, i.e. parameters like strings and booleans,
    where the elements cannot be ordered using lesser, equal, greater than
    (or at least it doesn't make sense ordering them like it doesn't make sense to order "true" and "false").

    Warning: Categorical parameters are treated as a sort of Ordinal parameters, this may not work in general.
    """

    def __init__(self, values, default, probability_distribution):
        """
        Initilization method. The possible values for this parameter are defined by the list values.
        :param values: list of possible values for this parameter.
        :param default: default value.
        :param probability_distribution: the string "uniform" for the uniform probability density function or a list of values describing the probability distribution.
        """
        Parameter.__init__(self)
        self.values = values
        self.default = default
        self.distribution = []
        if (
            probability_distribution == "uniform"
        ):  # In this case give equal probability to all possible parameter levels
            self.prior = "uniform"
            probability = 1.0 / self.get_discrete_size()
            for i in range(self.get_discrete_size()):
                self.distribution.append(probability)
        else:
            self.prior = "distribution"
            self.distribution = probability_distribution

    def randomly_select(self, size=1):
        """
        Select at random following the distribution given in the json.
        :return: a random number.
        :param size: the number of sampled random points
        """
        if self.get_prior() == "uniform":
            return np.random.choice(self.get_size(), size=size)
        else:
            return np.random.choice(self.get_size(), size=size, p=self.distribution)

    def randomly_select_uniform(self, size=1):
        """
        Sample from the prior distribution for this parameter.
        :param size: the number of sampled random points
        :return: the random sampled values from the set of available values.
        """
        return np.random.choice(self.get_size(), size=size)

    def pdf(self, x):
        """
        Compute the probability of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        distribution = self.get_parameter_distribution()
        return distribution[x]

    def get_x_probability(self, x):
        """
        Compute the probability of a given X under the prior distribution of the paramer.
        :return: the probability of X
        """
        # For categorical parameters, the pdf gives a probability and not a density.
        return self.pdf(x)

    def get_size(self):
        return len(self.values)

    def get_discrete_size(self):
        return self.get_size()

    def get_discrete_values(self):
        values = []
        for idx, val in enumerate(self.values):
            values.append(idx)
        return values

    def get_values(self):
        return self.values

    def get_int_values(self):
        return [self.get_int_value(value) for value in self.values]

    def get_default(self):
        if self.default == None:
            return None
        else:
            return self.values.index(self.default)

    def get_int_value(self, str_value):
        return self.values.index(str_value)

    def get_original_string(self, idx_value):
        return self.values[idx_value]

    def get_prior(self):
        return self.prior

    def get_parameter_distribution(self):
        return self.distribution


################# Space class ##################
class Space:
    def __init__(self, config):
        """
        Space is the class the defines the (input) search space.
        The type and sizes of input parameters are kept here.
        :param config: the json config.
        """
        max_number_of_predictions = config["max_number_of_predictions"]
        self.parameters_type = OrderedDict()
        self.parameters_python_type = OrderedDict()
        self.optimization_metrics = config["optimization_objectives"]
        self.timestamp_name = config["timestamp"]
        self.enable_feasible_predictor = ("feasible_output" in config) and (
            config["feasible_output"]["enable_feasible_predictor"] is True
        )
        if self.enable_feasible_predictor:
            feasible_output = config["feasible_output"]
            self.feasible_output_name = feasible_output["name"]
            self.feasible_output_true = feasible_output["true_value"]
            self.feasible_output_false = feasible_output["false_value"]
            self.parameters_type[self.feasible_output_name] = "feasible"
            self.parameters_python_type[
                self.feasible_output_name
            ] = "bool"  # The feasible output is always boolean (at least for now)
        else:
            self.feasible_output_name = None

        # Process input parameters from the json file
        self.all_input_parameters = OrderedDict()
        self.input_non_categorical_parameters = OrderedDict()
        self.input_categorical_parameters = OrderedDict()
        self.parse_input_parameters(config["input_parameters"])

        if self.get_discrete_space_size() > max_number_of_predictions:
            self.max_number_of_predictions = max_number_of_predictions
        else:
            self.max_number_of_predictions = self.get_discrete_space_size()
        for metric in self.optimization_metrics:
            self.parameters_type[metric] = "optimization_metric"
            self.parameters_python_type[
                metric
            ] = "float"  # Metrics are always float (at least for now)
        self.parameters_type[self.timestamp_name] = "timestamp"
        self.parameters_python_type[
            self.timestamp_name
        ] = "float"  # Timestamps are always floats
        self.output_metrics = self.get_output_parameters()
        self.input_output_parameter_names = (
            self.get_input_parameters() + self.output_metrics
        )
        self.time_metrics = self.get_timestamp_parameter()
        self.input_output_and_timestamp_parameter_names = (
            self.input_output_parameter_names + self.time_metrics
        )
        self.parameter_means = {}
        self.parameter_stds = {}
        self.normalize_inputs = config["normalize_inputs"]
        self.normalize_priors = False
        self.output_scalers = {}
        self.pdf = None
        self.bw_param = config["bandwidth_parameter"]
        self.bw_n_factor = config["bandwidth_n_factor"]
        # If we use weights, we have to adapt this to use "neff"
        self.bw_selector = lambda kde: np.power(
            kde.n * self.bw_n_factor, -1.0 / (kde.d + self.bw_param)
        )

        hypermapper_mode = config["hypermapper_mode"]["mode"]
        if hypermapper_mode == "exhaustive":
            self.exhaustive_search_file = deal_with_relative_and_absolute_path(
                "", config["hypermapper_mode"]["exhaustive_search_file"]
            )

        # If using multivariate kde priors
        self.estimate_prior_flag = config["estimate_multivariate_priors"]
        prior_estimation_data = None
        if self.estimate_prior_flag:
            self.normalize_priors = True
            prior_estimation_file = config["prior_estimation_file"]
            point_quantile = config["prior_estimation_quantile"]
            prior_estimation_data, _ = self.load_data_file(
                prior_estimation_file, only_valid=self.enable_feasible_predictor
            )
            all_input_data = np.array(
                [
                    self.convert_types_to_string(j, prior_estimation_data)
                    for j in list(self.get_input_parameters())
                ]
            )
            # We will have to adapt the priors for a multiobjective setting
            threshold = np.quantile(
                prior_estimation_data[self.optimization_metrics[0]], point_quantile
            )
            indices = np.nonzero(
                prior_estimation_data[self.optimization_metrics[0]] < threshold
            )[0]
            good_points = all_input_data[:, indices]
            self.pdf = gaussian_kde(good_points, bw_method=self.bw_selector)

        # If using univariate kde priors, define prior for each parameter
        for param_idx, param in enumerate(config["input_parameters"]):
            if config["input_parameters"][param]["prior"] == "estimate":
                self.normalize_priors = True
                if prior_estimation_data is None:
                    prior_estimation_file = config["prior_estimation_file"]
                    point_quantile = config["prior_estimation_quantile"]
                    prior_estimation_data, _ = self.load_data_file(
                        prior_estimation_file, only_valid=self.enable_feasible_predictor
                    )
                    # We will have to adapt the priors for a multiobjective setting
                    threshold = np.quantile(
                        prior_estimation_data[self.optimization_metrics[0]],
                        point_quantile,
                    )
                    indices_good = np.nonzero(
                        prior_estimation_data[self.optimization_metrics[0]] < threshold
                    )[0]
                for input_param in self.get_input_parameters():
                    if self.parameters_type[input_param] == "real":
                        good_points = np.array(prior_estimation_data[input_param])[
                            indices_good
                        ]
                        self.all_input_parameters[input_param].estimate_pdf(
                            good_points, bw_method=self.bw_selector
                        )
            elif config["input_parameters"][param]["prior"] == "custom_gaussian":
                self.normalize_priors = True
                if len(config["custom_gaussian_prior_means"]) == 1:
                    mean = config["custom_gaussian_prior_means"][0]
                elif len(config["custom_gaussian_prior_means"]) == len(
                    config["input_parameters"]
                ):
                    mean = config["custom_gaussian_prior_means"][param_idx]
                else:
                    print(
                        "Error: the custom_gaussian prior means array must be either 1 or",
                        len(config["input_parameters"]),
                        "received",
                        len(config["custom_gaussian_prior_means"]),
                    )
                    raise SystemExit
                if len(config["custom_gaussian_prior_stds"]) == 1:
                    std = config["custom_gaussian_prior_stds"][0]
                elif len(config["custom_gaussian_prior_stds"]) == len(
                    config["input_parameters"]
                ):
                    std = config["custom_gaussian_prior_stds"][param_idx]
                else:
                    print(
                        "Error: the custom_gaussian prior stds array must be either 1 or",
                        len(config["input_parameters"]),
                        "received",
                        len(config["custom_gaussian_prior_stds"]),
                    )
                    raise SystemExit
                self.all_input_parameters[param].set_custom_gaussian_prior_params(
                    mean, std
                )
            elif isinstance(config["input_parameters"][param]["prior"], str):
                self.normalize_priors = True

    def parse_input_parameters(self, input_parameters_json):
        """
        Parse the input search space from the json file.
        :param input_parameters_json: the property of the json file describing all the input parameters.
        :return: the data structures of self are initialized. This method doesn't return any other value.
        """
        for param_name, param in input_parameters_json.items():
            param_type = param["parameter_type"]
            param_default = None
            if "parameter_default" in param:
                param_default = param["parameter_default"]
            do_not_use_prior = (
                False  # Internal switch to not use the prior distribution in the json
            )
            if do_not_use_prior:
                param_distribution = "uniform"
            else:
                param_distribution = param["prior"]
            if param_type == "ordinal":
                param_values = param["values"]
                self.all_input_parameters[param_name] = OrdinalParameter(
                    values=param_values,
                    default=param_default,
                    probability_distribution=param_distribution,
                )
                self.input_non_categorical_parameters[
                    param_name
                ] = self.all_input_parameters[param_name]
                float_flag = 0
                for number in param_values:
                    if isinstance(number, float):
                        float_flag = 1
                self.parameters_type[param_name] = "ordinal"
                if float_flag == 0:
                    self.parameters_python_type[param_name] = "int"
                else:
                    self.parameters_python_type[param_name] = "float"
            elif param_type == "categorical":
                param_values = param["values"]
                if isinstance(param_distribution, str):
                    if param_distribution != "uniform":
                        print(
                            "Error in the json file: the values of parameter %s must be a valid probability distribution, i.e. only uniform or a list of probabilities is valid for categorical parameters. Exit."
                            % param_name
                        )
                        exit()
                else:
                    cumulative_probabilty = 0.0
                    for probability in param_distribution:
                        cumulative_probabilty += probability
                    if len(param_distribution) != len(param_values):
                        print(
                            "Error in the json file: the number of distribution probabilities has to be equal to the number of levels in the categorical parameter for parameter %s. Exit."
                            % param_name
                        )
                        exit()
                    if cumulative_probabilty != 1:
                        print(
                            "Error in the json file: the sum of distribution probabilities has to be 1 for parameter %s. Exit"
                            % param_name
                        )
                        exit()
                self.all_input_parameters[param_name] = CategoricalParameter(
                    values=param_values,
                    default=param_default,
                    probability_distribution=param_distribution,
                )
                self.input_categorical_parameters[
                    param_name
                ] = self.all_input_parameters[param_name]
                self.parameters_type[param_name] = "categorical"
                self.parameters_python_type[param_name] = "int"
            elif param_type == "real":
                param_min, param_max = param["values"]
                param_discretization = np.linspace(param_min, param_max, num=10)
                self.all_input_parameters[param_name] = RealParameter(
                    min_value=param_min,
                    max_value=param_max,
                    preferred_discretization=param_discretization,
                    default=param_default,
                    probability_distribution=param_distribution,
                )
                self.input_non_categorical_parameters[
                    param_name
                ] = self.all_input_parameters[param_name]
                self.parameters_type[param_name] = "real"
                self.parameters_python_type[param_name] = "float"
            elif param_type == "integer":
                param_min, param_max = param["values"]
                if param_min > param_max:
                    param_min, param_max = (
                        param_max,
                        param_min,
                    )  # breaks if min is greater than max. It does not break for real variables.
                self.all_input_parameters[param_name] = IntegerParameter(
                    min_value=param_min,
                    max_value=param_max,
                    default=param_default,
                    probability_distribution=param_distribution,
                )
                self.input_non_categorical_parameters[
                    param_name
                ] = self.all_input_parameters[param_name]
                self.parameters_type[param_name] = "integer"
                self.parameters_python_type[param_name] = "int"

    def get_type(self, parameter):
        """
        Get the HyperMapper type of the parameter given as a input, i.e. "ordinal", "categorical", "integer" or "real".
        :param parameter: the parameter we want to get the HyperMapper type from.
        :return: a string indicating the HyperMapper type.
        """
        if parameter in self.parameters_type:
            return self.parameters_type[parameter]
        else:
            print("Error: type not found.")
            exit(1)

    def get_python_type(self, parameter):
        """
        Get the python type of the parameter given as a input.
        :param parameter: the parameter we want to get the python type from.
        :return: a string indicating the python type.
        """
        if parameter in self.parameters_python_type:
            return self.parameters_python_type[parameter]
        else:
            print("Error: type not found.")
            exit(1)

    def get_input_parameters(self):
        """
        Get the parameters of input previously declared in the json file.
        :return: a list of strings of input parameters.
        """
        return list(self.all_input_parameters.keys())

    def get_input_parameters_objects(self):
        """
        Get the parameters of input previously declared in the json file in the form of a dictionary of objects.
        Each object describes one parameter, e.g. type, range of values.
        :return: a dictionary of parameters objects.
        """
        return self.all_input_parameters

    def get_input_categorical_parameters(self, input_parameters_names_list):
        """
        Get the categorical parameter names of input previously declared in the json file.
        :param input_parameters_names_list: a list containing the names of the input parameters to be retrieved.
        This act as a filter, applying this method only on the parameters listed in this variable.
        :return: a list of strings of input parameters.
        """
        return_parameters = []
        input_categorical_parameters = list(self.input_categorical_parameters.keys())
        for param in input_parameters_names_list:
            if param in input_categorical_parameters:
                return_parameters.append(param)
        return return_parameters

    def get_input_categorical_parameters_objects(self, input_parameters_names_list):
        """
        Get the all the categorical parameters of input previously declared in the json file in the form of a dictionary of objects.
        Each object describes one parameter, e.g. type, range of values.
        :param input_parameters_names_list: a list containing the names of the input parameters to be retrieved.
        This act as a filter, applying this method only on the parameters listed in this variable.
        :return: a dictionary of parameters objects filtered on input_parameters_names_list.
        """
        return_parameters = {}
        input_categorical_parameters = self.input_categorical_parameters
        for param in input_parameters_names_list:
            if param in input_categorical_parameters.keys():
                return_parameters[param] = input_categorical_parameters[param]
        return return_parameters

    def get_input_non_categorical_parameters(self, input_parameters_names_list):
        """
        Get the non-categorical parameter names of input previously declared in the json file.
        :param input_parameters_names_list: a list containing the names of the input parameters to be retrieved.
        This act as a filter, applying this method only on the parameters listed in this variable.
        :return: a list of strings of input parameters.
        """
        return_parameters = []
        input_non_categorical_parameters = list(
            self.input_non_categorical_parameters.keys()
        )
        for param in input_parameters_names_list:
            if param in input_non_categorical_parameters:
                return_parameters.append(param)
        return return_parameters

    def get_input_non_categorical_parameters_objects(self, input_parameters_names_list):
        """
        Get all the non-categorical parameters of input previously declared in the json file in the form of a dictionary of objects.
        Each object describes one parameter, e.g. type, range of values.
        :param input_parameters_names_list: a list containing the names of the input parameters to be retrieved.
        This act as a filter, applying this method only on the parameters listed in this variable.
        :return: a dictionary of parameters objects.
        """
        return_parameters = {}
        input_non_categorical_parameters = self.input_non_categorical_parameters
        for param in input_parameters_names_list:
            if param in input_non_categorical_parameters.keys():
                return_parameters[param] = input_non_categorical_parameters[param]
        return return_parameters

    def get_output_parameters(self):
        """
        Get the parameters of output declared in the json file, these are the optimization metrics and the feasible field.
        :return: a list of strings of output parameters.
        """
        if self.feasible_output_name is None:
            return self.get_optimization_parameters()
        else:
            return self.get_optimization_parameters() + [self.feasible_output_name]

    def get_input_optimization_parameters(self):
        """
        Get the parameters of the input and the optimization declared in the json file, these are the optimization metrics and the input parameters defining the space.
        :return: a list of strings of input and optimization parameters.
        """
        return self.get_input_parameters() + self.get_optimization_parameters()

    def get_input_and_output_parameters(self):
        """
        Get the input and output parameters defined in the json file (including feasibility flags).
        :return: a list of strings of input and output parameters.
        """
        return self.input_output_parameter_names

    def get_input_output_and_timestamp_parameters(self):
        """
        Get the input, output, and timestamp parameter names (including feasibility flags).
        :return: a list of strings of input and output parameters.
        """
        return self.input_output_and_timestamp_parameter_names

    def get_timestamp_parameter(self):
        """
        Get the timestamp parameter declared in the json file.
        :return: a list of one timestamp parameter.
        """
        return [self.timestamp_name]

    def get_optimization_parameters(self):
        """
        Get the optimization parameters of output declared in the json file, these are the optimization metrics only.
        :return: a list of strings of output parameters.
        """
        return self.optimization_metrics

    def get_feasible_parameter(self):
        """
        Get the feasible parameter declared in the json file.
        :return: a list of one feasible parameter.
        """
        return [self.feasible_output_name]

    def get_estimate_prior_flags(self):
        """
        Get the estimate prior flag, true if the priors for input parameters are being estimated with multivariate kde
        :return: true if at least one parameter has estimated priors.
        """
        return [self.estimate_prior_flag]

    def get_space_size(self):
        """
        Get the the size of the Cartesian product of the parameters of input previously declared in the json file.
        :return: an integer that is the size of the search space, can be "inf" if the space is infinite.
        """
        total = 1
        for p in self.get_input_parameters():
            total *= self.all_input_parameters[p].get_size()
        return total

    def get_discrete_space_size(self):
        """
        Get the size of the Cartesian product of the discrete parameters of input previously declared in the json file.
        :return: an int that is the size of the search space. This is never "inf".
        """
        total = 1
        for p in self.get_input_parameters():
            total *= self.all_input_parameters[p].get_discrete_size()
        return total

    def get_space_for_prediction(self):
        """
        Get the Cartesian product of the discrete parameters of input previously declared in the json file.
        If the space is bigger than max_number_of_predictions then a random sample of max_number_of_predictions is returned instead.
        :return: a list of configurations.
        """
        if self.get_discrete_space_size() > self.max_number_of_predictions:
            return self.get_random_space(self.max_number_of_predictions)
        else:
            return itertools.product(
                *[
                    self.all_input_parameters[param].get_discrete_values()
                    for param in self.all_input_parameters.keys()
                ]
            )

    def get_space(self):
        """
        Get the Cartesian product of the discrete parameters of input previously declared in the json file.
        Warning: use this function only if the space size is small (how small will depend from the machine used,
        perhaps smaller than 10,000,000 in any case).
        :return: a dictionary (on the keys of the input space) that contains a list of values for the different configurations.
        """
        itertool_cartesian_product = itertools.product(
            *[
                self.all_input_parameters[param].get_discrete_values()
                for param in self.all_input_parameters.keys()
            ]
        )
        dictionary_of_configurations = defaultdict(list)
        for conf in itertool_cartesian_product:  # conf here is a tuple
            conf_header_index = 0
            for header in self.get_input_parameters():
                dictionary_of_configurations[header].append(conf[conf_header_index])
                conf_header_index += 1
        return dictionary_of_configurations

    def get_random_space(self, number_of_random_configurations):
        """
        Get the number of random configuration samples specified in input.
        :param number_of_random_configurations: the number of random configurations to return.
        :return: a list of configurations.
        """
        return [
            self.get_tuple_random_configuration()
            for conf in range(number_of_random_configurations)
        ]

    def get_tuple_random_configuration(self):
        """
        :return: a random configuration from the input parameter space under the form of a tuple.
        """
        return tuple(
            self.all_input_parameters[k].randomly_select()
            for k in self.get_input_parameters()
        )

    def get_random_configuration(self, use_priors=True, size=1, return_as_array=False):
        """
        :param use_priors: whether the prior distributions of the parameters should be used for the sampling
        :param size: the number of sampled random points
        :param return_as_array: whether to return the list-of-dicts (False) for points or the numpy array (True)
        :return: a number of random configurations from the parameter space under the form of a dictionary, or the sampled array, shape (size, dims)
        """
        oversampling_factor = 5
        configurations = {}
        input_parameters = self.get_input_parameters()
        input_parameter_objects = self.get_input_parameters_objects()
        estimate_prior_flag = self.get_estimate_prior_flags()[0]
        if estimate_prior_flag and use_priors:
            samples = np.array([])
            # do the same logical process here as inside each sample
            while len(samples) < size:
                configuration = {}
                new_samples = self.pdf.resample(size * oversampling_factor).T
                # retrieves which points are valid among the i:th dimension, then gives true for all elements which are valid among all dimensions
                valid_indices = np.all(
                    [
                        (new_samples[:, i] < input_parameter_objects[k].get_max())
                        & (new_samples[:, i] > input_parameter_objects[k].get_min())
                        for i, k in enumerate(input_parameter_objects)
                    ],
                    axis=0,
                )
                if len(samples) == 0:
                    samples = new_samples
                else:
                    samples = np.append(samples, new_samples[(valid_indices)], axis=0)
            # if the sampling gave us too many points, we just retrieve the 'size' first ones
            samples = samples[0:size]

        else:
            samples = np.array([[None] * len(input_parameters)] * size)
            for i, k in enumerate(input_parameters):
                if use_priors:
                    samples[:, i] = self.all_input_parameters[k].randomly_select(
                        size=size
                    )
                else:
                    samples[:, i] = self.all_input_parameters[
                        k
                    ].randomly_select_uniform(size=size)

        # in quick_RS, the samples are needed in numpy format to be able to efficiently remove duplicates
        if return_as_array:
            return samples

        if size == 1:
            for i, key in enumerate(input_parameters):
                configurations[key] = samples[0][i]
        else:
            # restructures the configurations in the way used throughout hypermapper - this line is the timesink in this process
            configurations = [
                {
                    key: samples[sample_nbr, dim]
                    for dim, key in enumerate(input_parameters)
                }
                for sample_nbr in range(len(samples[:, 0]))
            ]

        return configurations

    def get_configuration_probability(self, configuration):
        conf_list = np.array(
            [
                self.convert_types_to_string(j, configuration)
                for j in list(configuration.keys())
            ]
        )
        return self.pdf(conf_list)[0]

    def get_default_configuration(self):
        """
        :return: the default configuration from the input parameters space.
        """
        configuration = {}
        for k in list(self.all_input_parameters.keys()):
            configuration[k] = self.all_input_parameters[k].get_default()

        return configuration

    def get_default_or_random_configuration(self):
        """
        :return: the default configuration from the input parameters space.
        """
        configuration = self.get_default_configuration()

        if None in configuration.values():
            random_configuration = self.get_random_configuration()
            for param in configuration:
                if configuration[param] is None:
                    configuration[param] = random_configuration[param]

        return configuration

    def get_dimensions(self):
        """
        :return: the number of dimension of the input parameters space.
        """
        return len(self.all_input_parameters)

    def get_parameter_mean(self, parameter):
        """
        :param parameter: name of an input parameter.
        :return: the mean of the parameter requested
        """
        if parameter not in self.parameter_means.keys():
            print("Error: mean for", parameter, "not set")
            raise SystemExit
        return self.parameter_means[parameter]

    def get_parameter_std(self, parameter):
        """
        :param parameter: name of an input parameter.
        :return: the standard deviation of the parameter requested
        """
        if parameter not in self.parameter_stds.keys():
            print("Error: std for", parameter, "not set")
            raise SystemExit
        return self.parameter_stds[parameter]

    def set_parameter_mean(self, parameter, mean):
        """
        :param parameter: name of an input parameter.
        :param mean: mean of the input parameter.
        """
        self.parameter_means[parameter] = mean

    def set_parameter_std(self, parameter, std):
        """
        :param parameter: name of an input parameter.
        :param std: standard deviation of the input parameter.
        """
        self.parameter_stds[parameter] = std

    def get_input_normalization_flag(self):
        """
        :return: whether to normalize inputs or not.
        """
        return self.normalize_inputs

    def get_prior_normalization_flag(self):
        """
        :return: whether to normalize priors or not.
        """
        return self.normalize_priors

    def convert_strings_to_type(self, header, data_list):
        """
        From the parameter passed in the header variable, this function converts the list data_list from strings to that python data type.
        :param header: the parameter name.
        :param data_list: the list to convert.
        :return: a data_list converted in the python type of the parameter in header.
        """
        param_type = self.get_type(header)
        python_type = self.get_python_type(header)
        if param_type == "ordinal":
            if python_type == "int":
                return list(map(int, data_list))
            elif python_type == "float":
                return list(map(float, data_list))
        elif param_type == "real":
            return list(map(float, data_list))
        elif param_type == "integer":
            float_list = list(map(float, data_list))
            return list(map(int, float_list))
        elif param_type == "categorical":
            to_return = []
            for elem in data_list:
                to_return.append(self.all_input_parameters[header].get_int_value(elem))
            return to_return
        elif param_type == "optimization_metric":
            return list(map(float, data_list))
        elif param_type == "timestamp":
            return list(map(float, data_list))
        elif param_type == "feasible":
            if header == self.feasible_output_name:
                to_return = []
                for elem in data_list:
                    if elem == self.feasible_output_true:
                        to_return.append(True)
                    elif elem == self.feasible_output_false:
                        to_return.append(False)
                    else:
                        print(
                            "Error: parsing the feasible data, the value given in the input dataset doesn't correspond to the values declared in the json file."
                        )
                        exit(1)
                return to_return
            else:
                print("Error: wrong feasible keyword.")
                exit(1)
        else:
            print(
                "Error: type not found in the conversion between string and input data type."
            )
            exit(1)

    def convert_types_to_string(self, header, data_list):
        """
        From the parameter passed in the header variable, this function converts data_list in a list composed
        of values of the original type used by the user in the json file.
        This is useful when HyperMapper wants to write back to file the configurations found in the original type format than the json file.
        :param header: the parameter name.
        :param data_list: the list to convert.
        :return: a data_list converted in the original json file type.
        """
        param_type = self.get_type(header)
        python_type = self.get_python_type(header)
        if param_type == "ordinal":
            return data_list[header]
        elif param_type == "real":
            return data_list[header]
        elif param_type == "integer":
            return data_list[header]
        elif param_type == "categorical":
            to_return = []
            for elem in data_list[header]:
                to_return.append(
                    self.all_input_parameters[header].get_original_string(elem)
                )
            return to_return
        elif param_type == "optimization_metric":
            return data_list[header]
        elif param_type == "timestamp":
            return data_list[header]
        elif param_type == "feasible":
            if header == self.feasible_output_name:
                to_return = []
                for elem in data_list[header]:
                    if elem == True:
                        to_return.append(self.feasible_output_true)
                    elif elem == False:
                        to_return.append(self.feasible_output_false)
                    else:
                        print(
                            "Error: parsing the feasible data, the value given in the conversion to string function doesn't correspond to a valid value. It is: %s"
                            % str(elem)
                        )
                        exit(1)
                return to_return
            else:
                print("Error: wrong feasible keyword.")
                exit(1)
        else:
            print(
                "Error: type not found in the conversion between string and input data type."
            )
            exit(1)

    def load_data_file(
        self,
        data_file,
        debug=False,
        number_of_cpus=0,
        selection_keys_list=[],
        only_valid=False,
    ):
        """
        This function read data from a csv file.
        :param data_file: the csv file where the data to be loaded resides.
        :param debug: active if debugging mode is enabled.
        :param selection_keys_list: contains the key columns of the csv file to be filtered.
        :param number_of_cpus: not implemented yet (for future dev).
        :return: 2 variables: data_array which is the dictionary containing for each key an array of values of one input parameter;
        fast_addressing_of_data_array is an array that enables a fast addressing of data_array via a unique identifier string.
        """
        with open(data_file, "r") as f_csv:
            data = list(csv.reader(f_csv, delimiter=","))
        data = [i for i in data if len(i) > 0]

        start_time = datetime.datetime.now()
        data_array = {}
        fast_addressing_of_data_array = {}

        number_of_points = len(data) - 1
        if debug:
            print(
                (
                    "The number of points we are reading from the file "
                    + data_file
                    + " is: "
                    + str(number_of_points)
                )
            )
            print(
                (
                    "The total number of rows contained by the file "
                    + data_file
                    + " is: "
                    + str(len(data))
                )
            )

        headers = data[0]  # The first row contains the headers
        headers = [header.strip() for header in headers]

        parameter_names = self.input_output_parameter_names

        # Check correctness
        for parameter_name in parameter_names:
            if parameter_name not in headers:
                print(
                    "Error: when reading the input dataset file the following entry was not found in the dataset but declared as a input/output parameter: %s"
                    % parameter_name
                )
                exit(1)

        if self.timestamp_name in headers:
            parameter_names = self.input_output_and_timestamp_parameter_names

        for parameter_name in parameter_names:
            data_array[parameter_name] = []

        col_list = list(
            zip(*data)
        )  # Transform the data from a list of rows in a list of columns

        for parameter_name in parameter_names:
            i = 100000  # Random init, unimportant
            idx = 0
            for header in headers:
                if parameter_name == header:
                    i = idx
                else:
                    idx += 1
            data_array[parameter_name] = self.convert_strings_to_type(
                parameter_name, col_list[i][1:]
            )

        # Filtering the valid rows
        if only_valid:
            data_array_clean = {}

            for parameter in parameter_names:
                data_array_clean[parameter] = []

            for configuration in range(len(data_array[self.feasible_output_name])):
                flag_negative_objective_is_invalid = False
                if data_array[self.feasible_output_name][configuration] == True:
                    for parameter in parameter_names:
                        data_array_clean[parameter].append(
                            data_array[parameter][configuration]
                        )

            data_array = data_array_clean

        for index in range(len(data_array[self.output_metrics[0]])):
            configuration = {}
            for k in self.get_input_parameters():
                configuration[k] = data_array[k][index]
            str_data = self.get_unique_hash_string_from_values(configuration)
            if str_data in fast_addressing_of_data_array:
                print(
                    "Warning: duplicate configuration found "
                    + str_data
                    + " when loading the file: "
                    + data_file
                    + ", at line: "
                    + str(index + 2)
                )
            fast_addressing_of_data_array[str_data] = index

        print(
            (
                "Time to read from file "
                + data_file
                + " and create the data array is: "
                + str((datetime.datetime.now() - start_time).total_seconds())
                + " sec"
            )
        )

        # Just in case we are filtering on the columns:
        if selection_keys_list == []:
            return data_array, fast_addressing_of_data_array
        else:
            selected_dict = {
                parameter: data_array[parameter] for parameter in selection_keys_list
            }
            return selected_dict, fast_addressing_of_data_array

    def remove_duplicate_configs(self, *configs, ignore_columns=None):
        """
        Removes the duplicates from the combined configurations configs, and lets the first configs keep the remaining
        configurations from the duplicates
        :param configs: the configurations to be checked for duplicates - duplicates are checked across all configurations, with the first occurance being kept
        :ignore_column: don't consider the entered columns when checking for duplicates
        :return: the configurations with duplicates removed
        """
        full_config = np.concatenate(configs, axis=0).astype(float)
        if ignore_columns is not None:
            full_config = np.delete(full_config, ignore_columns, axis=1)
        _, unique_index = np.unique(full_config, return_index=True, axis=0)
        # index where every array in the concatenated array starts, plus where the last one ends
        config_lengths = np.array([len(config) for config in configs])
        config_lengths = np.insert(config_lengths, 0, 0)
        start_indices = np.cumsum(config_lengths)
        reduced_indices = np.array([None] * len(configs))
        for i in range(len(configs)):
            reduced_indices[i] = (
                unique_index[
                    (unique_index >= start_indices[i])
                    & (unique_index < start_indices[i + 1])
                ]
                - start_indices[i]
            )

        reduced_configs = [
            config[reduced_index]
            for config, reduced_index in zip(configs, reduced_indices)
        ]
        return reduced_configs

    def load_data_files(self, files, selection_keys_list=[], only_valid=False):
        """
        Create a new data structure that contains the merged info from all the files.
        :param files: the input files that we want to merge.
        :return: an array with the info in the param files merged.
        """
        data_array = {}
        for filename in files:
            (
                local_data_array,
                local_data_array_fast_addressing_of_data_array,
            ) = self.load_data_file(
                filename, selection_keys_list=selection_keys_list, only_valid=only_valid
            )
            data_array = concatenate_data_dictionaries(
                local_data_array, data_array, selection_keys_list
            )
        return data_array

    def get_unique_hash_string_from_values(self, configuration):
        """
        Returns a string that identifies a configuration in an unambiguous way.
        :param configuration: dictionary containing the pairs key/value.
        :return: a string representing a unique identifier to the given configuration.
        """
        str_data = ""
        configuration_input_parameters = configuration.keys()
        for parameter in self.get_input_parameters():
            if (
                parameter in configuration_input_parameters
            ):  # Double check that the parameter is available in configuration. There are cases in the code where we pass a configuration that only contains a subset of parameters of the original search space but we still want to compute an unique string for those cases.
                element = configuration[parameter]
                if isinstance(element, float) and element.is_integer():
                    element = int(
                        element
                    )  # ensure consistency in how numbers are represented in the string
                str_element = str(configuration[parameter]) + "_"
            else:
                str_element = "" + "_"
            str_data += str_element
        return str_data

    def random_sample_array(
        self, all_data_array, beginning_of_time, number_of_RS=1000, debug=False
    ):
        """
        This function is used in the case where exhaustive search is available.
        Random samples a dictionary where the keys are the columns of data samples. Each sample reads from the arrays at the same row.
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :param all_data_array: the dictionary arrays to sample.
        :param output_metrics: these are the optimization metrics and the feasible field names.
        :param input_parameters: these are the input feature names.
        :param number_of_RS: number of total random samples without repetitions to perform.
        :param debug: is the debug flag enabled?
        :return: 2 outputs: the dictionary containing the data sampled and a fast addressing of the same data in the form of an array.
        """
        input_parameters = self.get_input_parameters()
        len_all_data_array = len(all_data_array[input_parameters[0]])
        if debug:
            start_time = datetime.datetime.now()
            print(
                (
                    "The number of points we are reading from the data array is: "
                    + str(number_of_RS)
                )
            )
            print(
                (
                    "The total number of rows contained in the data array is: "
                    + str(len_all_data_array)
                )
            )

        data_array = {}
        fast_addressing_of_data_array = {}
        for var in self.input_output_and_timestamp_parameter_names:
            data_array[var] = []

        random.seed(0)
        integer_random_samples = self.random_no_repeat(
            range(len_all_data_array), number_of_RS
        )
        for index in range(len(integer_random_samples)):
            configuration_index = integer_random_samples[index]

            configuration = {}
            for k in input_parameters:
                configuration[k] = all_data_array[k][configuration_index]

            if self.isConfigurationAlreadyRun(
                fast_addressing_of_data_array, configuration
            ):
                print(
                    "Warning: unexpected duplication in the random samples. This sample is discarded (which will reduce the total # of random samples). Execution will not be interrupted."
                )
                continue

            for var in self.input_output_parameter_names:
                data_array[var].append(all_data_array[var][configuration_index])
            data_array[self.time_metrics[0]].append(
                self.current_milli_time() - beginning_of_time
            )

            str_data = self.get_unique_hash_string_from_values(configuration)
            fast_addressing_of_data_array[str_data] = index

        if debug:
            print(
                (
                    "Time to RS the data_array is: "
                    + str((datetime.datetime.now() - start_time).total_seconds())
                    + " sec"
                )
            )

        return data_array, fast_addressing_of_data_array

    def random_sample_configurations_without_repetitions(
        self, fast_addressing_of_data_array, number_of_RS, use_priors=True
    ):
        """
        Get a list of number_of_RS configurations with no repetitions and that are not already present in fast_addressing_of_data_array.
        :param fast_addressing_of_data_array: configurations previously selected.
        :param number_of_RS: the number of unique random samples needed.
        :return: a list of dictionaries. Each dictionary represents a configuration.
        """
        configurations = []
        alreadyRunRandom = 0
        RS_configurations_count = 0

        arr = list(fast_addressing_of_data_array.values())
        if len(arr) == 0:
            absolute_configuration_index = 0
        else:
            absolute_configuration_index = np.asarray(arr, dtype=np.int).max() + 1

        # See if input space is big enough otherwise it doesn't make sense to draw number_of_RS samples.
        if (self.get_space_size() - len(fast_addressing_of_data_array)) <= number_of_RS:
            configurations_aux = self.get_space()

            tmp_configurations = (
                self.filter_already_run_and_fill_with_random_configurations(
                    fast_addressing_of_data_array, configurations_aux, 0
                )
            )
            for conf_index in range(
                len(tmp_configurations[self.get_input_parameters()[0]])
            ):
                configuration = {}
                for header in self.get_input_parameters():
                    configuration[header] = tmp_configurations[header][conf_index]
                configurations.append(configuration)
        else:
            while RS_configurations_count != number_of_RS:
                configuration = self.get_random_configuration(use_priors)

                if self.isConfigurationAlreadyRun(
                    fast_addressing_of_data_array, configuration
                ):
                    alreadyRunRandom += 1
                    if alreadyRunRandom <= 1000000:
                        continue  # pick another configuration
                    else:
                        print(
                            "\n ####\n Warning: reached maximum number of Random sampling that have been already run. \nThe Random sampling configuration selection will stop now. Is the search space very small?\n"
                        )
                        break  # too many random samples failed, probably the space is very small

                str_data = self.get_unique_hash_string_from_values(configuration)
                fast_addressing_of_data_array[str_data] = absolute_configuration_index
                absolute_configuration_index += 1

                configurations.append(configuration)
                RS_configurations_count += 1

        return configurations

    def standard_latin_hypercube_sampling_configurations_without_repetitions(
        self,
        fast_addressing_of_data_array,
        number_of_samples,
        input_parameters_names_list,
    ):
        """
        Standard latin hypercube sampling (Standard LHS) techniques like in 2012 SANDIA Surrogate models for mixed
        discrete-continuous variables (Swiler et al, 2012).
        m is (2.2.9) in the paper.
        m*n is number_of_samples => n = number_of_samples/m.
        This procedure works also in the case of categorical parameters.
        :param fast_addressing_of_data_array: configurations previously selected.
        :param number_of_samples: the number of unique LHS samples needed.
        :param input_parameters_names_list: a list containing the names of the input parameters.
        This act as a filter, applying this method only on the parameters listed in this variable.
        :return: a set of configurations following the standard latin hypercube sampling algorithm.
        """
        from pyDOE import lhs

        tmp_configurations = []
        m = 1  # m is the size of the Cartesian product of the categorical variables.
        parameters_values_categorical = {}
        for key, value in self.get_input_categorical_parameters_objects(
            input_parameters_names_list
        ).items():
            m = m * value.get_size()  # Cartesian product size
            parameters_values_categorical[key] = value.get_int_values()
            # This shuffling is useful when we don't have enough budget to deal with the whole Cartesian product m.
            # In this case we want to randomize the choice of which configuration is selected.
            shuffle(parameters_values_categorical[key])

        # Sample using the latin hypercube sampling algorithm. lhs returns values from the [0, 1] interval.
        lhs_samples = lhs(
            len(self.get_input_non_categorical_parameters(input_parameters_names_list)),
            samples=number_of_samples,
        )

        # Scale values of non-categorical variables from [0, 1] to actual parameter values.
        # If a distribution for the parameter is defined in the json it takes into consideration the distribution.
        X = []
        for param_index, param_object in enumerate(
            self.get_input_non_categorical_parameters_objects(
                input_parameters_names_list
            ).values()
        ):
            if param_object.prior == "distribution":
                # This is the case of a distribution (distribution instead of a density) ordinal.
                object_distribution = param_object.get_parameter_distribution()
                object_values = param_object.get_values()
                # Using the indices instead of directly object_values with the rv_discrete is a workaround.
                # For some reason rv_discrete.ppf() returns float values while this is not always what is in object_
                # values (it is integers sometimes). To not change the original type we use the trick of using indices.
                object_indices = range(0, len(object_values), 1)
                param_distribution = stats.rv_discrete(
                    values=(object_indices, object_distribution)
                )
                # distribution = stats.rv_discrete(values=(param_object.get_values(), param_object.get_parameter_distribution()))
                aux = np.asarray(lhs_samples[:, param_index])
                aux2 = param_distribution.ppf(aux)
                aux2 = aux2.astype(int)
                aux3 = np.asarray(object_values)[aux2]
                X.append(list(aux3))
                # X.append(list(distribution.ppf(lhs_samples[:, param_index]))) # The ppf here maps from the probability value pk in [0,1] to the xk defined in the distribution.
            else:
                a = param_object.densities_alphas[param_object.prior]
                b = param_object.densities_betas[param_object.prior]
                X.append(
                    param_object.from_range_0_1_to_parameter_value(
                        np.asarray(stats.beta(a, b).ppf(lhs_samples[:, param_index]))
                    )
                )

        # Filling up the sampled configurations with the non-categorical parameters first.
        for i in range(len(X[0])):
            configuration = {}
            for j, param_name in enumerate(
                self.get_input_non_categorical_parameters(input_parameters_names_list)
            ):
                configuration[param_name] = X[j][i]
            tmp_configurations.append(configuration)

        # Dealing with categorical parameters
        exploit_prior_information = True
        # The algorithm that doesn't exploit the prior information is the one introduced by Swiler et al., 2012.
        # They split the amount of samples equally for the categoricals.
        if exploit_prior_information:
            # In this part we exploit the prior information version of the lhs algorithm,
            # we compute a joint distribution on the categorical parameters and then we use this information to split
            # deterministically the samples.
            # Note that this approach is deterministic and not random even if based on the prior probability distribution.
            # Another approach would be to sample following the joint distribution and then having a probabilistic approach.
            # However this is contrary in spirit to the LHS algorithm that tries to fill the space more evenly than
            # random sampling.
            # This new way of splitting the categoricals is a change with respect to (Swiler et al, 2012) because
            # here we use the prior present in the json to sample.
            # In the case the prior is not present this algorithm this algorithm is equivalent to Swiler et al. 2012.

            # Compute the joint distribution.
            joint_distribution = np.zeros(shape=(m, 2))
            input_categorical_parameters_objects = [
                param_object
                for param_object in self.get_input_categorical_parameters_objects(
                    input_parameters_names_list
                ).values()
            ]
            cartesian_product = [
                cartesian_element
                for cartesian_element in itertools.product(
                    *[
                        param_values
                        for param_values in parameters_values_categorical.values()
                    ]
                )
            ]  # cartesian_element here is a tuple
            for level in range(m):
                joint = 1
                tuple_cartesian_product = cartesian_product[level]
                for i, tuple_i in enumerate(tuple_cartesian_product):
                    joint *= input_categorical_parameters_objects[
                        i
                    ].get_parameter_distribution()[tuple_i]
                joint_distribution[level][0] = level
                joint_distribution[level][1] = joint

            df_joint_distribution = pd.DataFrame(
                joint_distribution, columns=["level", "joint"]
            )
            df_joint_distribution.sort_values(by=["joint"], ascending=[0], inplace=True)

            counter = 0
            #            for param_index, cartesian_element in enumerate(cartesian_product):
            for index_cartesian_product in range(len(cartesian_product)):
                number_of_samples_per_level = int(
                    math.floor(
                        number_of_samples
                        * df_joint_distribution["joint"].iloc[index_cartesian_product]
                    )
                )
                for index_number_of_samples_per_level in range(
                    number_of_samples_per_level
                ):
                    for j, param in enumerate(parameters_values_categorical.keys()):
                        tmp_configurations[counter][param] = cartesian_product[
                            int(
                                df_joint_distribution["level"].iloc[
                                    index_cartesian_product
                                ]
                            )
                        ][j]
                    counter += 1
                    if (
                        counter >= number_of_samples
                    ):  # Not enough sampling budget to continue on the whole Cartesian product of size m
                        break
                if (
                    counter >= number_of_samples
                ):  # Not enough sampling budget to continue on the whole Cartesian product of size m
                    break

            # This deals with the reminder fill up (loop tail) case where the number n doesn't allow to cover all the samples requested.
            if counter < number_of_samples:
                for cartesian_element in itertools.product(
                    *[
                        param_values
                        for param_values in parameters_values_categorical.values()
                    ]
                ):
                    for j, param in enumerate(parameters_values_categorical.keys()):
                        tmp_configurations[counter][param] = cartesian_element[j]
                    counter += 1
                    if counter >= number_of_samples:
                        break

        else:
            # Compute the amount of the split among the levels of the categorical variables.
            # The max with 1 deals with the corner case where the number of samples is strictly less than the Cartesian product m.
            # In that case the algo will create as many configurations as possible following the LHS algorithm but
            # we won't be able to split the sampling according to the whole m.
            n = max(1, math.floor(number_of_samples / m))

            # Which is, splitting the configurations on the levels of the categorical parameters like explained in (Swiler et al, 2012).
            counter = 0
            for cartesian_element in itertools.product(
                *[
                    param_values
                    for param_values in parameters_values_categorical.values()
                ]
            ):  # cartesian_element here is a tuple
                for i in range(n):
                    for j, param in enumerate(parameters_values_categorical.keys()):
                        tmp_configurations[counter][param] = cartesian_element[j]
                    counter += 1
                    if (
                        counter >= number_of_samples
                    ):  # No enough sampling budget to continue on the whole Cartesian product of size m
                        break
                if (
                    counter >= number_of_samples
                ):  # No enough sampling budget to continue on the whole Cartesian product of size m
                    break

            # This deals with the reminder fill up (loop tail) case where the number n doesn't allow to cover all the samples requested.
            if counter < number_of_samples:
                for cartesian_element in itertools.product(
                    *[
                        param_values
                        for param_values in parameters_values_categorical.values()
                    ]
                ):
                    for j, param in enumerate(parameters_values_categorical.keys()):
                        tmp_configurations[counter][param] = cartesian_element[j]
                    counter += 1
                    if counter >= number_of_samples:
                        break

        # Check that all the configurations are unique.
        # That is true in general (by definition of the LHS algorithm) but
        # since in the ordinal parameters case we compress range of values in one value this becomes not true anymore.
        configurations = []
        absolute_configuration_index = len(fast_addressing_of_data_array)
        duplicate_configurations = 0
        for configuration in tmp_configurations:
            str_data = self.get_unique_hash_string_from_values(configuration)
            if self.isConfigurationAlreadyRun(
                fast_addressing_of_data_array, configuration
            ):
                print(
                    "Warning: duplicate configuration found %s, replacing with a random sample."
                    % str(configuration)
                )
                duplicate_configurations += 1
                continue
            fast_addressing_of_data_array[str_data] = absolute_configuration_index
            configurations.append(configuration)
            absolute_configuration_index += 1

        # Complete the missing samples (i.e. the duplicate samples that have been eliminated) with random samples.
        added_configurations = self.random_sample_configurations_without_repetitions(
            fast_addressing_of_data_array, duplicate_configurations
        )
        for configuration in added_configurations:
            configurations.append(configuration)

        return configurations

    def k_latin_hypercube_sampling_configurations_without_repetitions(
        self, fast_addressing_of_data_array, number_of_samples, input_parameters_objects
    ):
        """
        k latin hypercube sampling (k LHS) techniques like in 2012 SANDIA Surrogate models for mixed discrete-continuous variables (Swiler et al, 2012).
        m is (2.2.9) in the paper.
        m*n is number_of_samples => n = number_of_samples/m.
        This procedure works also in the case of categorical parameters.
        :param fast_addressing_of_data_array: configurations previously selected.
        :param number_of_samples: the number of unique LHS samples needed.
        :param input_parameters_objects: the dictionary containing the objects representing all the input parameters
        :return: a set of configurations following the k latin hypercube sampling algorithm.
        """
        configurations = []
        m = 1
        input_parameters_categorical = []
        parameters_values_categorical = {}
        input_parameters_objects_non_categorical = {}
        for key, value in input_parameters_objects.items():
            if type(value) is CategoricalParameter:
                m = m * value.get_size()
                input_parameters_categorical.append(key)
                parameters_values_categorical[key] = value.get_int_values()
                shuffle(
                    parameters_values_categorical[key]
                )  # This shuffling is useful when we don't have enough budget to deal with the whole Cartesian product m. In this case we want to randomize the choice of which configuration is selected.
            else:
                input_parameters_objects_non_categorical[key] = value

        n = max(
            1, math.floor(number_of_samples / m)
        )  # The max with 1 deals with the corner case where the number of samples is strictly less than m. In that case the algo will create as many configurations as possible following the k LHS algorithm but we won't be able to split the sampling according to the whole Cartesian product m.
        # Dealing with the categorical parameters, which is, computing m separate lhs, one for each level of the permutations of the categorical parameters like explained in (Swiler et al, 2012).
        counter = 0
        for cartesian_element in itertools.product(
            *[param_values for param_values in parameters_values_categorical.values()]
        ):  # cartesian_element here is a tuple
            tmp_fast_addressing_of_data_array = {}
            tmp_configurations = self.standard_latin_hypercube_sampling_configurations_without_repetitions(
                tmp_fast_addressing_of_data_array,
                n,
                input_parameters_objects_non_categorical,
            )
            for i, configuration in enumerate(tmp_configurations):
                configurations.append(configuration)
                for j, param in enumerate(parameters_values_categorical.keys()):
                    configurations[counter][param] = cartesian_element[j]
                counter += 1
                if (
                    counter >= number_of_samples
                ):  # No enough sampling budget to continue on the whole Cartesian product of size m
                    break
            if (
                counter >= number_of_samples
            ):  # No enough sampling budget to continue on the whole Cartesian product of size m
                break

        # Update fast_addressing_of_data_array with configurations
        absolute_configuration_index = len(fast_addressing_of_data_array)
        for configuration in configurations:
            str_data = self.get_unique_hash_string_from_values(configuration)
            fast_addressing_of_data_array[str_data] = absolute_configuration_index
            absolute_configuration_index += 1

        # Complete the missing samples (i.e. the reminder of the number_of_samples/m division) with random samples.
        reminder_samples = max(0, number_of_samples - counter)
        reminder_configurations = self.random_sample_configurations_without_repetitions(
            fast_addressing_of_data_array, reminder_samples
        )
        for configuration in reminder_configurations:
            configurations.append(configuration)

        return configurations

    def grid_search(self, fast_addressing_of_data_array):
        configurations = []
        discrete_space = self.get_space()

        absolute_configuration_index = len(fast_addressing_of_data_array)
        for idx in range(self.get_discrete_space_size()):
            configuration = {}
            for param in discrete_space.keys():
                configuration[param] = discrete_space[param][idx]
            str_data = self.get_unique_hash_string_from_values(configuration)
            if str_data not in fast_addressing_of_data_array:
                fast_addressing_of_data_array[str_data] = absolute_configuration_index
                absolute_configuration_index += 1
                configurations.append(configuration)
        return configurations

    def get_doe_sample_configurations(
        self, fast_addressing_of_data_array, number_of_samples, doe_type
    ):
        """
        Get a list of number_of_samples configurations with no repetitions and that are not already present in fast_addressing_of_data_array.
        The configurations are sampled following the design of experiments (DOE) in the doe input variable.
        :param fast_addressing_of_data_array: configurations previously selected.
        :param number_of_samples: the number of unique samples needed.
        :param doe_type: type of design of experiments (DOE) chosen.
        :return: a list of dictionaries. Each dictionary represents a configuration.
        """
        configurations = []
        alreadyRunRandom = 0
        configurations_count = 0

        arr = list(fast_addressing_of_data_array.values())
        if len(arr) == 0:
            absolute_configuration_index = 0
        else:
            absolute_configuration_index = np.asarray(arr, dtype=np.int).max() + 1

        # See if input space is big enough otherwise it doesn't make sense to draw number_of_samples samples.
        if (
            self.get_space_size() - len(fast_addressing_of_data_array)
        ) <= number_of_samples:
            configurations_aux = self.get_space()

            tmp_configurations = (
                self.filter_already_run_and_fill_with_random_configurations(
                    fast_addressing_of_data_array, configurations_aux, 0
                )
            )
            for conf_index in range(
                len(tmp_configurations[self.get_input_parameters()[0]])
            ):
                configuration = {}
                for header in self.get_input_parameters():
                    configuration[header] = tmp_configurations[header][conf_index]
                configurations.append(configuration)
        else:
            if doe_type == "random sampling":
                configurations = self.random_sample_configurations_without_repetitions(
                    fast_addressing_of_data_array, number_of_samples
                )
            elif doe_type == "standard latin hypercube":
                input_parameters_objects = self.get_input_parameters_objects()
                configurations = self.standard_latin_hypercube_sampling_configurations_without_repetitions(
                    fast_addressing_of_data_array,
                    number_of_samples,
                    input_parameters_objects,
                )
            elif doe_type == "k latin hypercube":
                input_parameters_objects = self.get_input_parameters_objects()
                configurations = (
                    self.k_latin_hypercube_sampling_configurations_without_repetitions(
                        fast_addressing_of_data_array,
                        number_of_samples,
                        input_parameters_objects,
                    )
                )
            elif doe_type == "sliced latin hypercube":
                print(
                    "Error: sliced latin hypercube sampling not implemented yet. Exit."
                )
                exit()
            elif doe_type == "grid_search":
                configurations = self.grid_search(fast_addressing_of_data_array)
            else:
                print("Error: design of experiment sampling method not found. Exit.")
                exit()

        return configurations

    def run_configurations_client_server(
        self,
        beginning_of_time,
        configurations,
        run_directory,
        dontrun=False,
        doSleep=False,
        number_of_cpus=0,
    ):
        """
        Run a set of configurations in client-server mode under the form of a list of configurations.
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :param configurations: a list of configurations (dict).
        :param run_directory: the working directory where HyperMapper is run (this is specified in the json file).
        :param dontrun:
        :param doSleep:
        :return:
        """
        debug = False
        new_data_array = defaultdict(list)
        input_parameters = self.get_input_parameters()
        file_to_receive_from_interactive_system = deal_with_relative_and_absolute_path(
            run_directory, "interactive_protocol_file.csv.out"
        )

        # Remove file from the hard disk to prevent reading multiple times the same file
        if not debug:
            import os

            try:
                os.remove(file_to_receive_from_interactive_system)
            except OSError:
                pass

        print("Communication protocol: sending message...")
        read_write_on_a_file = False  # This is the default case where we want communication via file instead of stdin/out
        if read_write_on_a_file:
            file_to_send_to_interacting_system = deal_with_relative_and_absolute_path(
                run_directory, "interactive_protocol_file.csv"
            )
            # Write to standard output to communicate with the interacting system,
            # this is part of the Hypermapper/interacting system protocol
            with open(file_to_send_to_interacting_system, "w") as f:
                w = csv.writer(f)
                w.writerow([parameter for parameter in input_parameters])

                for conf_index in range(len(configurations)):
                    str_data = []
                    for parameter in list(input_parameters):
                        # Need this line because the convert_types_to_string takes a dictionary with each entry being a list and return a list
                        configuration = {
                            parameter: [configurations[conf_index][parameter]]
                        }
                        str_data.append(
                            str(
                                (
                                    self.convert_types_to_string(
                                        parameter, configuration
                                    )
                                )[0]
                            )
                        )
                    w.writerow(str_data)
            sys.stdout.write_protocol(
                "FRequest %d %s\n"
                % (len(configurations), file_to_send_to_interacting_system)
            )

            if not debug:
                # Wait until the response is ready
                print("Waiting response from the third party software.")
                line = sys.stdin.readline()
                sys.stdout.write(line)
                ack_return_message = (
                    "Ready " + file_to_receive_from_interactive_system + "\n"
                )
                if line != ack_return_message:
                    print(
                        "Error: expecting '%s' and received '%s'. Exit."
                        % (ack_return_message, line)
                    )
                    exit(1)
            new_data_array, fast_addressing_of_data_array = self.load_data_file(
                file_to_receive_from_interactive_system
            )
        else:
            # Write to stdout
            sys.stdout.write_protocol(
                "Request %d\n" % len(configurations)
            )  # From the Hypermapper/interacting system protocol
            str_header = ""
            for parameter in list(input_parameters):
                str_header += str(parameter) + ","
            str_header = str_header[:-1] + "\n"
            sys.stdout.write_protocol(str_header)

            for conf_index in range(len(configurations)):
                str_data = ""
                for parameter in list(input_parameters):
                    # Need this line because the convert_types_to_string takes a dictionary with each entry being a list and return a list
                    configuration = {parameter: [configurations[conf_index][parameter]]}
                    str_data += (
                        str((self.convert_types_to_string(parameter, configuration))[0])
                        + ","
                    )
                str_data = str_data[:-1] + "\n"
                sys.stdout.write_protocol(str_data)

            if debug:
                # Just for testing without the need to run the system that is interacting with HyperMapper on the other side.
                (
                    new_data_array,
                    fast_addressing_of_data_array_interacting_system,
                ) = self.load_data_file(file_to_receive_from_interactive_system)
            else:
                # Read from stdin
                print("Communication protocol: receiving message....")
                line = sys.stdin.readline()
                sys.stdout.write(line)
                parameters_header = [x.strip() for x in line.split(",")]
                parameters_index_reference = {}
                for header in self.get_input_and_output_parameters():
                    if header not in parameters_header:
                        print(
                            "Key Error while getting the random configurations results."
                        )
                        print(
                            "The key HyperMapper was looking for is: %s" % str(header)
                        )
                        print("The headers received are: %s" % str(parameters_header))
                        print(
                            "The input-ouput parameters specified in the json are: %s"
                            % str(self.input_output_and_timestamp_parameter_names)
                        )
                        exit()
                    parameters_index_reference[header] = parameters_header.index(header)

                parameters_data = []
                for conf_index in range(len(configurations)):
                    line = sys.stdin.readline()
                    sys.stdout.write(line)
                    parameters_data.append([x.strip() for x in line.split(",")])

                parameters_data = list(
                    map(list, zip(*parameters_data))
                )  # Transpose the list of lists
                for header in self.get_input_and_output_parameters():
                    try:
                        new_data_array[header] = self.convert_strings_to_type(
                            header, parameters_data[parameters_index_reference[header]]
                        )
                    except ValueError as ve:
                        print("Failed to parse received message:")
                        print(ve)
                        raise SystemError
                new_data_array[self.get_timestamp_parameter()[0]] = [
                    self.current_milli_time()
                ] * len(configurations)
        if debug:
            print(
                "The size of the new set of samples in the run configurations method is %d"
                % len(next(iter(new_data_array.values())))
            )
        return new_data_array

    def run_configurations_from_data_array(
        self,
        all_data_array,
        all_fast_addressing_of_data_array,
        beginning_of_time,
        configurations,
        doSleep=False,
    ):
        """
        Run a specific configuration.
        This is the exhaustive mode (from data file mode), i.e. it reads from the array that has been previously created from the exhaustive search file.
        All the *minimum_execution* are done (TBD).
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :param all_data_array:
        :param all_fast_addressing_of_data_array:
        :param configurations:
        :param doSleep:
        :return:
        """
        new_data_array = defaultdict(list)
        for conf_index in range(len(configurations)):
            str_data = self.get_unique_hash_string_from_values(
                configurations[conf_index]
            )

            try:
                index = all_fast_addressing_of_data_array[str_data]
            except:
                print(
                    "Error: the new configuration to be run has not been found in the csv file.",
                    str_data,
                )
                exit()

            for parameter in list(all_data_array.keys()):
                new_data_array[parameter].append(all_data_array[parameter][index])
            new_data_array[self.time_metrics[0]].append(
                self.current_milli_time() - beginning_of_time
            )
        return new_data_array

    def _run_configurations_with_black_box_function(
        self, configurations, black_box_function, beginning_of_time, batch_mode=False
    ):
        """
        Run a list of configurations.
        This method was added for compatibility purposes: it decides whether to send all configurations to the black-box function at once or one at a time.
        compatibility option is sending them one at a time.
        :param configurations: a list of configurations (dict).
        :param black_box_function: objective function being optimized.
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :param batch_mode: whether the black-box function supports batch execution.
        :return:
        """
        if batch_mode:
            data_array = self.run_configurations_with_black_box_function(
                configurations, black_box_function, beginning_of_time
            )
        else:
            data_array = defaultdict(list)
            for configuration in configurations:
                tmp_data_array = self.run_configurations_with_black_box_function(
                    [configuration], black_box_function, beginning_of_time
                )
                for key in tmp_data_array:
                    data_array[key] += tmp_data_array[key]
        return data_array

    def run_configurations_with_black_box_function(
        self, configurations, black_box_function, beginning_of_time
    ):
        """
        Run a list of configurations.
        :param configurations: a list of configurations (dict).
        :param black_box_function: objective function being optimized.
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :return:
        """
        new_data_array = defaultdict(list)
        tmp_configurations = defaultdict(list)
        for configuration in configurations:
            if sorted(list(configuration.keys())) != sorted(
                self.get_input_parameters()
            ):
                print("Configuration does not match input parameters")
                print("Expected:", self.get_input_parameters())
                print("Got:", list(configuration.keys()))
                raise SystemExit
            tmp_configuration = {}
            for param in configuration:
                if self.get_type(param) == "categorical":
                    tmp_configuration[param] = self.all_input_parameters[
                        param
                    ].get_original_string(configuration[param])
                else:
                    tmp_configuration[param] = configuration[param]
                new_data_array[param].append(configuration[param])
                tmp_configurations[param].append(tmp_configuration[param])

        # For compatibility, we send a dictionary of numbers instead of a dictionary of lists if we have only one configuration
        if len(configurations) == 1:
            tmp_configurations = tmp_configuration

        objective_values = black_box_function(tmp_configurations)
        output_parameters = self.get_output_parameters()

        # if we have a single output_parameter, black-box return does not need to be a dictionary
        # convert it so that we always have a dictionary
        if len(output_parameters) == 1 and type(objective_values) is not dict:
            objective_values = {output_parameters[0]: objective_values}

        for output_param in output_parameters:
            if type(objective_values[output_param]) is list:
                if len(objective_values[output_param]) != len(configurations):
                    print("Error running black-box function:")
                    print(
                        f"Requested evaluation of {len(configurations)} configurations, but received {len(objective_values[output_param])} values instead."
                    )
                    raise SystemExit
                new_data_array[output_param] += objective_values[output_param]
            else:
                if len(configurations) > 1:
                    print("Error running black-box function:")
                    print(
                        f"Requested evaluation of multiple configurations, black box function should have returned a list. Returned {type(objective_values[output_param])} instead."
                    )
                    raise SystemExit
                new_data_array[output_param].append(objective_values[output_param])

        new_data_array[self.get_timestamp_parameter()[0]] = [
            self.current_milli_time() - beginning_of_time
        ] * len(configurations)
        return new_data_array

    def run_configurations(
        self,
        hypermapper_mode,
        configurations,
        beginning_of_time,
        black_box_function=None,
        exhaustive_search_data_array=None,
        exhaustive_search_fast_addressing_of_data_array=None,
        run_directory=None,
        number_of_cpus=0,
        batch_mode=False,
    ):
        """
        Run a set of configurations in one of HyperMappers modes.
        :param hypermapper_mode: which HyperMapper mode to run as.
        :param configurations: a list of configurations (dict).
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :param run_directory: the working directory where HyperMapper is run (this is specified in the json file).
        :param black_box_function: objective function being optimized.
        :param beginning_of_time: time from the beginning of the HyperMapper design space exploration.
        :param exhaustive_search_data_array: array containing all data from the exhaustive search.
        :param exhaustive_search_fast_addressing_of_data_array: dict containing the indices of each configuration in the data array.
        :param run_directory: the directory where HyperMapper is running.
        :param number_of_cpus: number of cpus to use in parallel.
        :return: dictionary with the new configurations that were evaluated
        """
        if hypermapper_mode == "default":
            if black_box_function is None:
                print("Error: a black box function is required in default mode")
                raise SystemExit
            data_array = self._run_configurations_with_black_box_function(
                configurations, black_box_function, beginning_of_time, batch_mode
            )
            self.print_data_array(data_array)
        elif hypermapper_mode == "exhaustive":
            print("Running on exhaustive mode.")
            if exhaustive_search_data_array is None:
                print("Error: missing exhaustive_search_data_array parameter")
                raise SystemExit
            if exhaustive_search_fast_addressing_of_data_array is None:
                print(
                    "Error: missing exhaustive_search_fast_addressing_of_data_array parameter"
                )
                raise SystemExit
            data_array = self.run_configurations_from_data_array(
                exhaustive_search_data_array,
                exhaustive_search_fast_addressing_of_data_array,
                beginning_of_time,
                configurations=configurations,
                doSleep=False,
            )
        elif hypermapper_mode == "client-server":
            print("Running on client-server mode.")
            if run_directory is None:
                print("Error: missing run_directory parameter")
                raise SystemExit
            data_array = self.run_configurations_client_server(
                beginning_of_time,
                configurations,
                run_directory,
                dontrun=False,
                doSleep=False,
                number_of_cpus=number_of_cpus,
            )
        else:
            print("Unrecognized hypermapper mode:", hypermapper_mode)
            raise SystemExit
        return data_array

    def print_data_array(self, data_array):
        """
        Print a data array in a csv format.
        :param data_array: the data array to print. A dict of lists.
        """
        keys = ""
        for key in data_array.keys():
            keys += str(key) + ","
        print(keys[:-1])

        for idx in range(len(data_array[list(data_array.keys())[0]])):
            configuration = ""
            for key in data_array.keys():
                # configuration += str(data_array[key][idx]) + ','
                configuration += (
                    str(self.convert_types_to_string(key, data_array)[idx]) + ","
                )
            print(configuration[:-1])
        print()

    def isConfigurationAlreadyRun(self, fast_addressing_of_data_array, configuration):
        """
        Function that returns True if the configuration is contained by the array data_array and False otherwise.
        :param fast_addressing_of_data_array:
        :param configuration:
        :return:
        """
        str_data = self.get_unique_hash_string_from_values(configuration)
        try:
            fast_addressing_of_data_array[str_data]
            return True
        except:
            return False

    def current_milli_time(self):
        return int(round(time.time() * 1000))

    def random_no_repeat(self, numbers, count):
        """
        Generates count numbers of non repetitive integers. Example:
        # import random
        # random.seed(0)
        # random_no_repeat(range(12), 10)
        [1, 9, 8, 5, 10, 2, 3, 7, 4, 0]
        """
        if len(numbers) < count:
            print(
                "Warning: the number of random samples without repetition requested is less than the total number of choices. Returning # of choices random samples."
            )
            count = len(numbers)
        number_list = list(numbers)
        random.shuffle(number_list)
        return number_list[:count]

    def filter_already_run_and_fill_with_random_configurations(
        self,
        fast_addressing_of_data_array,
        dictionary_to_be_filtered,
        runs_per_optimization_iteration,
    ):
        """
        Filter the dictionary dictionary_to_be_filtered with respect to the configurations that have already been run.
        :param fast_addressing_of_data_array: configurations previously selected.
        :param dictionary_to_be_filtered: the number of unique random samples needed.
        :param runs_per_optimization_iteration: this is the maximum number of runs in one optimization iteration requested by the user.
        The returned number of configurations will be matching this number.
        :return: a dictionary that is filtered.
        """
        dictionary_to_be_returned = defaultdict(list)
        len_dictionary_to_be_filtered = len(
            dictionary_to_be_filtered[self.get_input_parameters()[0]]
        )
        for conf_counter in range(len_dictionary_to_be_filtered):

            configuration = {}
            for header in self.get_input_parameters():
                configuration[header] = dictionary_to_be_filtered[header][conf_counter]

            if self.isConfigurationAlreadyRun(
                fast_addressing_of_data_array, configuration
            ):
                sys.stdout.write_to_logfile(
                    "Configuration "
                    + str(configuration)
                    + " already run, found in the filter."
                )
                continue

            for header in self.get_input_parameters():
                dictionary_to_be_returned[header].append(configuration[header])

        len_configurations_to_run = len(
            dictionary_to_be_returned[self.get_input_parameters()[0]]
        )
        if len_configurations_to_run < runs_per_optimization_iteration:
            print(
                "Warning: there are no enough configurations to run (only %d, required %d). Filling with random sampling to keep optimization going..."
                % (len_configurations_to_run, runs_per_optimization_iteration)
            )
            number_of_RS = runs_per_optimization_iteration - len_configurations_to_run
            configurations = self.random_sample_configurations_without_repetitions(
                fast_addressing_of_data_array, number_of_RS
            )
            for configuration in configurations:
                for header in self.get_input_parameters():
                    dictionary_to_be_returned[header].append(configuration[header])

        for header in self.get_input_parameters():
            dictionary_to_be_returned[header] = np.asarray(
                dictionary_to_be_returned[header]
            )

        return dictionary_to_be_returned
