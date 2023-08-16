import sys
from typing import Dict, Any, Union

import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import GammaPrior, LogNormalPrior
from botorch.fit import fit_gpytorch_mll
import warnings

from hypermapper.bo.models.models import Model


class GpGpytorch(gpytorch.models.ExactGP, Model):
    """
    Class implementing our version of the gpytorch GP kernel.
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
        y = y.to(X)

        if not settings["noise"]:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-6))
        else:
            # define noise priors
            if settings["noise_prior"]["name"] == "gamma":
                alpha = float(settings["lengthscale_prior"]["parameters"][0])
                beta = float(settings["lengthscale_prior"]["parameters"][1])
                noise_prior = GammaPrior(concentration=alpha, rate=beta)

            elif settings["noise_prior"]["name"] == "lognormal":
                mu = float(settings["lengthscale_prior"]["parameters"][0])
                sigma = float(settings["lengthscale_prior"]["parameters"][1])
                noise_prior = LogNormalPrior(loc=mu, scale=sigma)

            else:
                noise_prior = None

            noise_constraint = GreaterThan(1e-8)

            # define the likelihood
            likelihood = GaussianLikelihood(
                noise_constraint=noise_constraint,
                noise_prior=noise_prior,
            )

        gpytorch.models.ExactGP.__init__(self, X, y, likelihood)

        if settings["lengthscale_prior"]["name"] == "gamma":
            alpha = float(settings["lengthscale_prior"]["parameters"][0])
            beta = float(settings["lengthscale_prior"]["parameters"][1])
            lengthscale_prior = GammaPrior(concentration=alpha, rate=beta)

        elif settings["lengthscale_prior"]["name"] == "lognormal":
            mu = float(settings["lengthscale_prior"]["parameters"][0])
            sigma = float(settings["lengthscale_prior"]["parameters"][1])
            lengthscale_prior = LogNormalPrior(loc=mu, scale=sigma)

        else:
            lengthscale_prior = None

            # define lengthscale constraints
        # if configuration.gp.lengthscale_constraint:
        #    if configuration.gp.lengthscale_constraint[1]:
        #        lengthscale_constraint = Interval(*configuration.gp.lengthscale_constraint)
        #    else:
        #        lengthscale_constraint = GreaterThan(configuration.gp.lengthscale_constraint[0])
        # else:
        #    lengthscale_constraint = None

        """
        Outputscale priors and constraints
        """
        # define outputscale priors
        if settings["outputscale_prior"]["name"] == "gamma":
            alpha = float(settings["outputscale_prior"]["parameters"][0])
            beta = float(settings["outputscale_prior"]["parameters"][1])
            outputscale_prior = GammaPrior(concentration=alpha, rate=beta)

        elif settings["outputscale_prior"]["name"] == "lognormal":
            mu = float(settings["outputscale_prior"]["parameters"][0])
            sigma = float(settings["outputscale_prior"]["parameters"][1])
            outputscale_prior = LogNormalPrior(loc=mu, scale=sigma)

        else:
            outputscale_prior = None

        """
        Initialise the kernel
        """
        # self.mean_module = gpytorch.means.ZeroMean()
        self.mean_module = gpytorch.means.ConstantMean()

        self.ard_size = X.shape[-1]

        base_kernel = gpytorch.kernels.MaternKernel(
            # lengthscale_constraint=lengthscale_constraint,
            lengthscale_prior=lengthscale_prior,
            ard_num_dims=self.ard_size,
            nu=2.5,
        )

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
            # outputscale_constraint=outputscale_constraint,
        )

    def forward(self, x: torch.tensor):
        """
        Evaluates mean and varaiance at a single point x.

        Input:
            x: input to evaluate.

        Returns:
            a MultivariateNormal object describing the probability distribution of the output.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def set_data(self, X, y):
        """
        updates the training data for the GP model.
        Input:
            - X: new x training data
            - y: new y training data
        """
        self.set_train_data(X, y, False)

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
            - previous_hyperparameters: hyperparameters from previous iterations
        Returns:
            - Hyperparameters of the model or None if the model is not fitted.
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        if settings["multistart_hyperparameter_optimization"]:
            worst_log_likelihood = np.inf
            best_log_likelihood = -np.inf
            best_GP = None

            n_iterations = settings["hyperparameter_optimization_iterations"]

            # gen sample points
            sample_points = [(self.covar_module.base_kernel.lengthscale, self.covar_module.outputscale,
                              self.likelihood.noise_covar.noise.item(), self.mean_module.constant.item())] + [
                                (10 ** (1.5 * np.random.random(self.train_inputs[0].shape[1]) - 1),
                                 10 ** (1.5 * np.random.random() - 1),
                                 10 ** (4 * np.random.random() - 5),
                                 0
                                 )
                                for _ in range(n_iterations)
                            ]

            for i, sample_point in enumerate(sample_points):
                try:
                    self.apply_hyperparameters(sample_point[0], sample_point[1], sample_point[2], sample_point[3])
                    try:
                        warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)
                        fit_gpytorch_mll(mll)
                    except Exception as e:
                        print(f"Warning: failed to fit in iteration {i}")
                        print(e)

                    mll_val = mll(self(*self.train_inputs), self.train_targets)
                    if mll_val > best_log_likelihood:
                        best_log_likelihood = mll_val
                        best_GP = (self.covar_module.base_kernel.lengthscale, self.covar_module.outputscale,
                                   self.likelihood.noise_covar.noise.item(), self.mean_module.constant.item())

                    if mll_val < worst_log_likelihood:
                        worst_log_likelihood = mll_val
                except Exception as e:
                    print(e)
                    pass

            if best_GP is None:
                print(f"Failed to fit the GP hyperparameters in all of the {settings['multistart_hyperparameter_optimization_iterations']} iterations.")
                sys.stdout.write_to_logfile(f"Failed to fit the GP hyperparameters in all of the {settings['multistart_hyperparameter_optimization_iterations']} iterations.\n")
                return None
            try:
                self.apply_hyperparameters(best_GP[0], best_GP[1], best_GP[2], best_GP[3])
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
                print(f"Warning: failed to fit model.")
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
        """
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            loss = -mll(self(*self.train_inputs), self.train_targets)
            loss.backward()
            optimizer.step()

    def get_mean_and_std(
            self,
            normalized_data,
            predict_noiseless,
            use_var=False
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
        prediction = self(normalized_data)
        mean = prediction.mean
        var = prediction.variance

        if any(var < -1e-12):
            raise Exception(f"GP prediction resulted in negative variance {var}")
        var += 1e-12

        if use_var:
            uncertainty = var
        else:
            uncertainty = np.sqrt(var)

        return mean, uncertainty
