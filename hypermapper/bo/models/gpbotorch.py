import sys
from typing import Dict, Any, Union

import botorch.models
import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior, LogNormalPrior
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
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
            and settings["noise_prior"]["name"].lower() == "gamma"
        ):
            noise_prior = GammaPrior(*settings["noise_prior"]["parameters"])
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        elif settings["noise_prior"] is not None and settings["noise_prior"]["name"].lower() == "lognormal":
            noise_prior = LogNormalPrior(*settings["noise_prior"]["parameters"])
            noise_prior_mode = None
        else:
            noise_prior = GammaPrior(1.1, 0.05)
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

        if settings["lengthscale_prior"]["name"].lower() == "gamma":
            alpha = float(settings["lengthscale_prior"]["parameters"][0])
            beta = float(settings["lengthscale_prior"]["parameters"][1])
            lengthscale_prior = GammaPrior(concentration=alpha, rate=beta)

        elif settings["lengthscale_prior"]["name"].lower() == "lognormal":
            mu = float(settings["lengthscale_prior"]["parameters"][0])
            sigma = float(settings["lengthscale_prior"]["parameters"][1])
            lengthscale_prior = LogNormalPrior(loc=mu, scale=sigma)

        else:
            lengthscale_prior = GammaPrior(3.0, 6.0)

        """
        Outputscale priors and constraints
        """
        # define outputscale priors
        if settings["outputscale_prior"]["name"].lower() == "gamma":
            alpha = float(settings["outputscale_prior"]["parameters"][0])
            beta = float(settings["outputscale_prior"]["parameters"][1])
            outputscale_prior = GammaPrior(concentration=alpha, rate=beta)

        elif settings["outputscale_prior"]["name"].lower() == "lognormal":
            mu = float(settings["outputscale_prior"]["parameters"][0])
            sigma = float(settings["outputscale_prior"]["parameters"][1])
            outputscale_prior = LogNormalPrior(loc=mu, scale=sigma)

        else:
            outputscale_prior = GammaPrior(2.0, 0.15)

        """
        Initialise the kernel
        """
        # mean_module = gpytorch.means.ZeroMean()
        mean_module = gpytorch.means.ConstantMean()

        self.ard_size = X.shape[-1]

        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=lengthscale_prior,
            ard_num_dims=self.ard_size,
            nu=2.5,
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
        )

        with warnings.catch_warnings(record=True) as w:  # This catches the non-normalized due to default outside of parameter range warning
            warnings.filterwarnings(
                "default", category=botorch.exceptions.InputDataWarning
            )
            botorch.models.SingleTaskGP.__init__(
                self,
                X,
                y.unsqueeze(1),
                likelihood=likelihood,
                covar_module=covar_module,
                mean_module=mean_module
            )
            for warning in w:
                sys.stdout.write_to_logfile(f"WARNING: {str(warning.message)}\n")

        self.eval()

    def apply_hyperparameters(self, lengthscale, outputscale, noise, mean):
        if not (type(lengthscale) is torch.Tensor and type(outputscale) is torch.Tensor and type(noise) is torch.Tensor and type(mean) is torch.Tensor):
            raise TypeError("Hyperparameters must be torch tensors")
        self.covar_module.base_kernel.lengthscale = lengthscale.to(dtype=torch.float64)
        self.covar_module.outputscale = outputscale.to(dtype=torch.float64)
        self.likelihood.noise_covar.noise = noise.to(dtype=torch.float64)
        if isinstance(self.mean_module, gpytorch.means.ConstantMean):
            self.mean_module.constant = mean.to(dtype=torch.float64)

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

        # warnings.filterwarnings(
        #     "ignore", category=gpytorch.utils.warnings.GPInputWarning
        # )
        self.train()
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
                    self.likelihood.noise_covar.noise.data,
                    (self.mean_module.constant.data if isinstance(self.mean_module, gpytorch.means.ConstantMean) else torch.tensor(0))
                )
            ] + [
                (
                    10 ** (1.5 * torch.rand(self.train_inputs[0].shape[1]) - 1),
                    10 ** (1.5 * torch.rand(1) - 1),
                    10 ** (4 * torch.rand(1) - 5),
                    torch.tensor(0)
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

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        fit_gpytorch_mll(mll)
                        for warning in w:
                            sys.stdout.write_to_logfile(f"{str(warning.message)}\n")
                    self.train(), self.likelihood.train()

                    mll_val = mll(self(*self.train_inputs), self.train_targets)
                    if mll_val > best_log_likelihood:
                        best_log_likelihood = mll_val
                        best_GP = (
                            self.covar_module.base_kernel.lengthscale,
                            self.covar_module.outputscale,
                            self.likelihood.noise_covar.noise.data,
                            (self.mean_module.constant.data if isinstance(self.mean_module, gpytorch.means.ConstantMean) else torch.tensor(0)),
                        )

                    if mll_val < worst_log_likelihood:
                        worst_log_likelihood = mll_val
                except Exception as e:
                    sys.stdout.write_to_logfile(
                        f"Warning: failed to fit in iteration {i}\n"
                    )
                    sys.stdout.write_to_logfile(f"{e}\n")

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
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    fit_gpytorch_mll(mll)
                    for warning in w:
                        sys.stdout.write_to_logfile(f"{str(warning.message)}\n")
            except Exception as e:
                sys.stdout.write_to_logfile("Warning: Failed to fit model.\n")
                sys.stdout.write_to_logfile(f"{e}\n")
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
            "mean": (self.mean_module.constant.data if isinstance(self.mean_module, gpytorch.means.ConstantMean) else torch.tensor(0))
        }
        self.eval()
        return hyperparameters

    def _backup_fit(self, mll):
        """
        Fits the model hyperparameters if the botorch LFBGS fit fails.

        Input:
            - mll: Marginal log likelihood.
        """
        mll.train()
        fit_gpytorch_mll(
            mll=mll,
            optimizer=fit_gpytorch_mll_torch
        )
        mll.eval()

    def get_mean_and_std(self, normalized_data, predict_noiseless):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: not used for this model.
            - use_var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        if predict_noiseless:
            prediction = self.posterior(normalized_data, observation_noise=False)
        else:
            prediction = self.posterior(normalized_data, observation_noise=True)

        mean = prediction.mean.reshape(-1)
        var = prediction.variance.reshape(-1)
        if any(var < -1e12):
            raise Exception(f"GP prediction resulted in negative variance {var}")
        var += 1e-12

        std = torch.sqrt(var)

        return mean, std