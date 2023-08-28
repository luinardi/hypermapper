import sys
from typing import Dict, Any, Union

import botorch.models
import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior
from botorch.fit import fit_gpytorch_mll
import warnings

from hypermapper.bo.models.models import Model


class GpBotorch(botorch.models.SingleTaskGP, Model):
    """
    A wrapper for the botorch GP https://botorch.org/.
    """

    def __init__(
        self,
        settings,
        X: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        input:
            - settings: Run settings
            - X: x training data
            - Y: y training data
        """
        y = y.to(X)

        if (
            settings["noise_prior"] is not None
            and settings["noise_prior"]["name"] == "Gamma"
        ):
            noise_prior = GammaPrior(*settings["noise_prior"]["parameters"])
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=torch.Size(),
                noise_constraint=GreaterThan(
                    1e-4,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            likelihood = None
        botorch.models.SingleTaskGP.__init__(
            self, X, y.unsqueeze(1), likelihood=likelihood
        )

    def apply_hyperparameters(self, lengthscale, outputscale, noise, mean):
        self.covar_module.base_kernel.lengthscale = lengthscale
        self.covar_module.outputscale = outputscale
        self.likelihood.noise_covar.noise = noise
        self.mean_module.constant = mean

    def fit(
        self,
        settings: Dict[str, Any],
        previous_hyperparameters: Union[Dict[str, Any], None],
    ):
        """
        Fits the model hyperparameters.
        Input:
            - settings:
            - previous_hyperparameters: Hyperparameters of the previous model.
        Returns:
            - Hyperparameters of the model or None if the model is not fitted.
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        if settings["multistart_hyperparameter_optimization"]:
            worst_log_likelihood = np.inf
            best_log_likelihood = -np.inf
            best_GP = None

            # gen sample points
            sample_points = [
                (
                    self.covar_module.base_kernel.lengthscale,
                    self.covar_module.outputscale,
                    self.likelihood.noise_covar.noise.item(),
                    self.mean_module.constant.item(),
                )
            ] + [
                (
                    10 ** (1.5 * np.random.random(self.train_inputs[0].shape[1]) - 1),
                    10 ** (1.5 * np.random.random() - 1),
                    10 ** (4 * np.random.random() - 5),
                    0,
                )
                for _ in range(settings["hyperparameter_optimization_iterations"])
            ]

            for i, sample_point in enumerate(sample_points):
                try:
                    self.apply_hyperparameters(
                        sample_point[0],
                        sample_point[1],
                        sample_point[2],
                        sample_point[3],
                    )
                    try:
                        warnings.filterwarnings(
                            "ignore", category=gpytorch.utils.warnings.GPInputWarning
                        )
                        fit_gpytorch_mll(mll)
                    except Exception as e:
                        print(f"Warning: failed to fit in iteration {i}")
                        print(e)

                    mll_val = mll(self(*self.train_inputs), self.train_targets)
                    if mll_val > best_log_likelihood:
                        best_log_likelihood = mll_val
                        best_GP = (
                            self.covar_module.base_kernel.lengthscale,
                            self.covar_module.outputscale,
                            self.likelihood.noise_covar.noise.item(),
                            self.mean_module.constant.item(),
                        )

                    if mll_val < worst_log_likelihood:
                        worst_log_likelihood = mll_val
                except Exception as e:
                    print(e)
                    pass

            if best_GP is None:
                sys.stdout.write_to_logfile(
                    f"Failed to fit the GP hyperparameters in all of the {settings['hyperparameter_optimization_iterations']} initial points.\n"
                )
                return None
            try:
                self.apply_hyperparameters(
                    best_GP[0], best_GP[1], best_GP[2], best_GP[3]
                )
            except np.linalg.LinAlgError as e:
                sys.stdout.write_to_logfile(
                    f"Caught exception when getting GP hyperparameters: {e}. Continuing.\n"
                )
            sys.stdout.write_to_logfile(
                f"Best log-likelihood: {best_log_likelihood} Worst log-likelihood: {worst_log_likelihood}\n"
            )
        else:
            mll = ExactMarginalLogLikelihood(self.likelihood, self)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print("Warning: Failed to fit model.")
                print(e)
                self._backup_fit(mll)

        sys.stdout.write_to_logfile(
            f"lengthscales:\n{self.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()}\n"
        )
        sys.stdout.write_to_logfile(
            f"kernel variance: {self.covar_module.outputscale.squeeze().detach().numpy()}\n"
        )
        sys.stdout.write_to_logfile(
            f"noise variance: {self.likelihood.noise_covar.noise.squeeze().detach().numpy()}\n"
        )
        hyperparameters = {
            "lengthscale": self.covar_module.base_kernel.lengthscale,
            "variance": self.covar_module.outputscale,
            "noise": self.likelihood.noise_covar.noise,
            "mean": self.mean_module.constant,
        }
        self.eval()
        return hyperparameters

    def _backup_fit(self, mll):
        """
        Fits the model hyperparameters if the botorch LFBGS fit fails.

        Input:
            - mll: Marginal log likelihood.
        """
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            loss = -mll(self(*self.train_inputs), self.train_targets)
            loss.backward()
            optimizer.step()

    def get_mean_and_std(self, normalized_data, predict_noiseless, use_var=False):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: not used for this model.
            - use_var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        prediction = self(normalized_data)
        mean = prediction.mean
        var = prediction.variance
        if any(var < -1e12):
            raise Exception(f"GP prediction resulted in negative variance {var}")
        var += 1e-12

        if use_var:
            uncertainty = var
        else:
            uncertainty = torch.sqrt(var)

        return mean, uncertainty


class GpBotorchHeteroskedastic(botorch.models.HeteroskedasticSingleTaskGP, Model):
    """
    Wrapper for Botorch heteroskedastic GP model. Their implementation is currently not working, so this is WIP.
    """

    def __init__(
        self,
        settings,
        X: torch.Tensor,
        y: torch.Tensor,
        std_estimate: torch.Tensor,
    ):
        """
        input:
            - settings: Run settings
            - X: x training data
            - Y: y training data
            - std_estimate: estimated standard deviation of the noise
        """
        y = y.to(X)
        yVar = std_estimate.to(X) ** 2
        botorch.models.HeteroskedasticSingleTaskGP.__init__(
            self, X, y.unsqueeze(1), yVar.unsqueeze(1)
        )

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
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print("Warning: Failed to fit model.")
            print(e)
            self._backup_fit(mll)

        sys.stdout.write_to_logfile(
            f"lengthscales:\n{self.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()}\n"
        )
        sys.stdout.write_to_logfile(
            f"kernel variance: {self.covar_module.outputscale.squeeze().detach().numpy()}\n"
        )
        hyperparameters = {
            "lengthscale": self.covar_module.base_kernel.lengthscale,
            "variance": self.covar_module.outputscale,
            "noise": 0,
            "mean": self.mean_module.constant,
        }
        self.eval()
        return hyperparameters

    def _backup_fit(self, mll):
        """
        Fits the model hyperparameters if the botorch LFBGS fit fails.
        """
        self.train()
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            out = self(*self.train_inputs)
            loss = -mll(out, self.train_targets)
            loss.backward()
            optimizer.step()

        self.eval()

    def get_mean_and_std(self, normalized_data, predict_noiseless, use_var=False):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: ignore noise when calculating variance
            - var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        prediction = self(normalized_data)
        mean = prediction.mean
        var = prediction.variance
        if any(var < -1e12):
            raise Exception(f"GP prediction resulted in negative variance {var}")
        var += 1e-12

        if use_var:
            uncertainty = var
        else:
            uncertainty = torch.sqrt(var)

        return mean, uncertainty


class GpBotorchFixed(botorch.models.FixedNoiseGP, Model):
    """
    Class implementing our version of the gpytorch GP kernel.
    """

    def __init__(
        self,
        settings,
        X: torch.Tensor,
        y: torch.Tensor,
        std_estimate: torch.Tensor,
    ):
        """
        input:
            - settings: Run settings
            - X: x training data
            - Y: y training data
        """
        y = y.to(X)
        yVar = std_estimate.to(X) ** 2
        botorch.models.FixedNoiseGP.__init__(self, X, y.unsqueeze(1), yVar.unsqueeze(1))

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
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print("Warning: Failed to fit model.")
            print(e)
            self._backup_fit(mll)

        sys.stdout.write_to_logfile(
            f"lengthscales:\n{self.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()}\n"
        )
        sys.stdout.write_to_logfile(
            f"kernel variance: {self.covar_module.outputscale.squeeze().detach().numpy()}\n"
        )
        hyperparameters = {
            "lengthscale": self.covar_module.base_kernel.lengthscale,
            "variance": self.covar_module.outputscale,
            "noise": 0,
            "mean": self.mean_module.constant,
        }
        self.eval()
        return hyperparameters

    def _backup_fit(self, mll):
        """
        Fits the model hyperparameters if the botorch LFBGS fit fails.
        """
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            loss = -mll(self(*self.train_inputs), self.train_targets)
            loss.backward()
            optimizer.step()

    def get_mean_and_std(self, normalized_data, predict_noiseless, use_var=False):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: ignore noise when calculating variance
            - var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        prediction = self(normalized_data)
        mean = prediction.mean
        var = prediction.variance
        if any(var < -1e12):
            raise Exception(f"GP prediction resulted in negative variance {var}")
        var += 1e-12

        if use_var:
            uncertainty = var
        else:
            uncertainty = np.sqrt(var)

        return mean, uncertainty
