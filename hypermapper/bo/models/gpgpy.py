import copy
import sys
from typing import Dict, Any, Union

import GPy
import numpy as np
import torch

from hypermapper.bo.models.models import Model


class GpGpy(GPy.models.GPRegression, Model):
    """
    Class implementing our version of the GPy GP kernel.
    """

    def __init__(
            self,
            settings: Dict[str, Any],
            X: torch.Tensor,
            y: torch.Tensor,
    ):
        """
        input:
            - settings: Run settings
            - X: x training data
            - Y: y training data
        """
        X = X.numpy()
        y = y.reshape(-1, 1).numpy()
        super(GpGpy, self).__init__(
            X,
            y,
            kernel=GPy.kern.Matern52(X.shape[1], ARD=True),
            normalizer=False,
        )

        if settings["normalize_inputs"]:
            if settings["lengthscale_prior"]["name"] == "gamma":
                alpha = float(settings["lengthscale_prior"]["parameters"][0])
                beta = float(settings["lengthscale_prior"]["parameters"][1])
                self.kern.lengthscale.set_prior(
                    GPy.priors.Gamma(alpha, beta)
                )
            elif settings["lengthscale_prior"]["name"] == "lognormal":
                mu = float(settings["lengthscale_prior"]["parameters"][0])
                sigma = float(settings["lengthscale_prior"]["parameters"][1])
                self.kern.lengthscale.set_prior(
                    GPy.priors.LogGaussian(mu, sigma)
                )
            if settings["outputscale_prior"]["name"] == "gamma":
                alpha = float(settings["outputscale_prior"]["parameters"][0])
                beta = float(settings["outputscale_prior"]["parameters"][1])
                self.kern.variance.set_prior(
                    GPy.priors.Gamma(alpha, beta)
                )
            if settings["noise_prior"]["name"] == "gamma":
                alpha = float(settings["noise_prior"]["parameters"][0])
                beta = float(settings["noise_prior"]["parameters"][1])
                self.likelihood.variance.set_prior(
                    GPy.priors.Gamma(alpha, beta)
                )
        if not settings["noise"]:
            self.likelihood.variance = 1e-6
            self.likelihood.fix()

    def set_data(self, X, y):
        """
        updates the training data for the GP model.
        Input:
            - X: new x training data
            - X: new y training data
        Returns:
            - Hyperparameters or None if the model is not trained
        """
        X = X.numpy()
        y = y.reshape(-1, 1).numpy()
        self.set_XY(X=X, Y=y)

    def fit(
            self,
            settings: Dict[str, Any],
            previous_hyperparameters: Union[Dict[str, Any], None],
    ):
        """
        Fits the model hyperparameters.
        Input:
            - settings:
            - previous_hyperparameters: hyperparameters from previous iterations
        Returns:
            - Hyperparameters of the model or None if the model is not fitted.
        """
        with np.errstate(
                divide="ignore", over="ignore", invalid="ignore"
        ):  # GPy's optimize has uncaught warnings that do not affect performance, suppress them so that they do not propagate to Hypermapper

            # if the initial lengthscales are small and the input space distances too large,
            # the Gram matrix becomes the identity matrix, and the optimizer fails completely
            while np.min(self.kern.K(self.X)) < 1e-10:
                sys.stdout.write_to_logfile(
                    f"Warning: initial lengthscale too short. Multiplying with 2. Highest similarity: {np.min(self.kern.K(self.X))}\n"
                )
                self.kern.lengthscale = (
                        self.kern.lengthscale * 2
                )
            if settings["multistart_hyperparameter_optimization"]:
                worst_log_likelihood = np.inf
                best_log_likelihood = -np.inf
                best_GP = None

                n_iterations = settings["hyperparameter_optimization_iterations"]

                # gen sample points
                sample_points = [
                    (10 ** (2 * np.random.random(len(self.kern.lengthscale)) - 1),
                     10 ** (2 * np.random.random() - 1),
                     10 ** (3 * np.random.random() - 5),
                     )
                    for _ in range(n_iterations)
                ]

                if settings["reuse_gp_hyperparameters"] and previous_hyperparameters:
                    sample_points.append(
                        (
                            tuple(previous_hyperparameters["lengthscale"]),
                            previous_hyperparameters["variance"],
                            previous_hyperparameters["noise"],
                        )
                    )

                # evaluate sample points
                sample_values = []
                for sample_point in sample_points:
                    try:
                        self.kern.lengthscale = sample_point[0]
                        self.kern.variance = sample_point[1]
                        self.likelihood.variance = sample_point[
                            2
                        ]
                        sample_values.append(
                            self._log_marginal_likelihood
                        )
                    except:
                        sample_values.append(-np.inf)
                best_initial_sample_points = [sample_points[i] for i in np.argpartition(sample_values, -n_iterations)[-n_iterations:]]
                for sample_point in best_initial_sample_points:
                    try:
                        self.kern.lengthscale = sample_point[0]
                        self.kern.variance = sample_point[1]
                        self.likelihood.variance = sample_point[
                            2
                        ]
                        self.optimize()

                        if self._log_marginal_likelihood > best_log_likelihood:
                            best_log_likelihood = self._log_marginal_likelihood
                            best_GP = self.to_dict()

                        if self._log_marginal_likelihood < worst_log_likelihood:
                            worst_log_likelihood = self._log_marginal_likelihood

                    except:
                        pass

                if best_GP is None:
                    sys.stdout.write_to_logfile(f"Failed to fit the GP hyperparameters in all of the {settings['hyperparameter_optimization_iterations']} iterations.\n")
                    return None
                try:
                    self.kern.lengthscale = best_GP['kernel']['lengthscale']
                    self.kern.variance = best_GP['kernel']['variance']
                    self.likelihood.variance = best_GP['likelihood']['variance']
                except np.linalg.LinAlgError as e:
                    sys.stdout.write_to_logfile(
                        f"Caught exception when getting GP hyperparameters: {e}. Continuing.\n"
                    )

                sys.stdout.write_to_logfile(
                    f"Best log-likelihood: {best_log_likelihood} Worst log-likelihood: {worst_log_likelihood}\n"
                )
            else:
                self.optimize()  # adding optimizer = 'scg' seems to yield slightly more stable lengthscales

            sys.stdout.write_to_logfile(
                f"lengthscales:\n{self.kern.lengthscale}\n"
            )
            sys.stdout.write_to_logfile(
                f"kernel variance:\n{self.kern.variance}\n"
            )
            sys.stdout.write_to_logfile(
                f"noise variance:\n{self.likelihood.variance}\n"
            )
            try:
                sys.stdout.write_to_logfile(
                    f"{self.kern.K(self.X)[:5, :5]}\n"
                )
            except Exception as e:
                print(e)

        hyperparameters = {
            "lengthscale": self.kern.lengthscale,
            "variance": self.kern.variance,
            "noise": self.likelihood.variance,
        }
        return hyperparameters

    def get_mean_and_std(
            self,
            normalized_data,
            predict_noiseless,
            use_var=False,
    ):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: ignore noise when calculating variance
            - use_var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        if predict_noiseless:
            mean, var = self.predict_noiseless(normalized_data.numpy())
        else:
            mean, var = self.predict(normalized_data.numpy())
        mean = mean.flatten()
        var = var.flatten()
        var[var < 10 ** -11] = 10 ** -11
        if use_var:
            uncertainty = var
        else:
            uncertainty = np.sqrt(var)

        return torch.tensor(mean), torch.tensor(uncertainty)
