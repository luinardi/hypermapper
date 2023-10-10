import datetime
import sys
import time
from abc import abstractmethod
from typing import Any, Tuple, List, Dict, Union

import torch

from hypermapper.bo.models.rf import RFRegressionModel, RFClassificationModel
from hypermapper.param.data import DataArray
from hypermapper.param.space import Space
from hypermapper.param.transformations import (
    transform_data,
    preprocess_parameters_array,
    transform_estimate,
)


class Model:
    @abstractmethod
    def fit(
        self,
        settings: Dict[str, Any],
        previous_hyperparameters: Union[Dict[str, Any], None],
    ):
        raise NotImplementedError

    @abstractmethod
    def get_mean_and_std(
        self,
        normalized_data,
        predict_noiseless
    ):
        raise NotImplementedError


def generate_mono_output_regression_models(
    settings: Dict[str, Any],
    data_array: DataArray,
    param_space: Space,
    objective_means: torch.Tensor = None,
    objective_stds: torch.Tensor = None,
    previous_hyperparameters: Dict[str, Union[float, List]] = None,
    reoptimize: bool = True,
) -> Union[Tuple[List[Any], Dict[str, float]], Tuple[None, None]]:
    """
    Fit a regression model, supported model types are Random Forest and Gaussian Process.
    This method fits one mono output model for each objective.

    Input:
        - settings: run settings
        - data_array: the data to use for training.
        - param_space: the parameter space
        - objective_means: means for the different objectives. Used for standardization.
        - objective_stds: stds for the different objectives. Used for standardization.
        - previous_hyperparameters: hyperparameters from the last trained GP.
        - reoptimize: if false, the model will just use previous HPs.

    Returns:
        - the regressors
        - hyperparameters
    """
    start_time = time.time()

    X, Y, parametrization_names = transform_data(
        settings, data_array, param_space, objective_means, objective_stds
    )

    models = []
    hyperparameters = None
    for i, metric in enumerate(param_space.metric_names):
        y = Y[:, i]
        if settings["models"]["model"] == "gaussian_process":
            if settings["GP_model"] == "gpy":
                from hypermapper.bo.models.gpgpy import GpGpy

                model = GpGpy(settings, X, y)
            elif settings["GP_model"] == "botorch":
                from hypermapper.bo.models.gpbotorch import GpBotorch

                model = GpBotorch(settings, X, y)
            elif settings["GP_model"] == "gpytorch":
                from hypermapper.bo.models.gpgpytorch import GpGpytorch

                model = GpGpytorch(settings, X, y)
            elif settings["GP_model"] == "botorch_fixed":
                from hypermapper.bo.models.gpbotorch import GpBotorchFixed

                std_estimate = transform_estimate(
                    settings, data_array.std_estimate, objective_means, objective_stds
                )
                model = GpBotorchFixed(settings, X, y, std_estimate)
            elif settings["GP_model"] == "botorch_heteroskedastic":
                from hypermapper.bo.models.gpbotorch import GpBotorchHeteroskedastic

                std_estimate = transform_estimate(
                    settings, data_array.std_estimate, objective_means, objective_stds
                )
                model = GpBotorchHeteroskedastic(settings, X, y, std_estimate)
            else:
                raise Exception("Unrecognized GP model type:", settings["GP_model"])
            if reoptimize:
                hyperparameters = model.fit(settings, previous_hyperparameters)
                if hyperparameters is None:
                    return None, None
            else:
                model.covar_module.base_kernel.lengthscale = tuple(
                    previous_hyperparameters["lengthscale"]
                )
                model.covar_module.outputscale = previous_hyperparameters["variance"]
                model.likelihood.noise_covar.noise = previous_hyperparameters["noise"]

        elif settings["models"]["model"] == "random_forest":
            model = RFRegressionModel(
                n_estimators=settings["models"]["number_of_trees"],
                max_features=settings["models"]["max_features"],
                bootstrap=settings["models"]["bootstrap"],
                min_samples_split=settings["models"]["min_samples_split"],
                use_all_data_to_fit_mean=settings["models"]["use_all_data_to_fit_mean"],
                use_all_data_to_fit_variance=settings["models"][
                    "use_all_data_to_fit_variance"
                ],
                add_linear_std=settings["models"]["add_linear_std"],
            )
            model.fit_rf(X, y)
        else:
            raise Exception("Unrecognized model type:", settings["models"]["model"])

        models.append(model)
    sys.stdout.write_to_logfile(
        ("End of training - Time %10.2f sec\n" % (time.time() - start_time))
    )
    return models, hyperparameters


def generate_classification_model(
    settings: Dict[str, Any],
    param_space: Space,
    data_array: DataArray,
) -> RFClassificationModel:
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices)).

    Input:
        - settings: run settings
        - param_space: parameter space object for the current application.
        - data_array: the data to use for training.
    Returns:
        - the classifier
    """
    start_time = datetime.datetime.now()
    X, names = preprocess_parameters_array(data_array.parameters_array, param_space)
    classifier = RFClassificationModel(
        settings,
        param_space,
        X,
        data_array.feasible_array,
    )

    sys.stdout.write_to_logfile(
        "End of training - Time %10.2f sec\n"
        % ((datetime.datetime.now() - start_time).total_seconds())
    )
    return classifier


def compute_model_mean_and_uncertainty(
    data: torch.Tensor,
    models: list,
    param_space: Space,
    predict_noiseless: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points.

    Input:
        - data: tensor containing points to predict.
        - models: models to use for the prediction.
        - param_space: parameter space object for the current application.
        - var: whether to compute variance or standard deviation.
        - predict_noiseless: ignore noise when calculating variance (only GP)

    Returns:
        - the predicted mean and uncertainty for each point.
    """
    X, names = preprocess_parameters_array(data, param_space)
    means = torch.Tensor()
    uncertainties = torch.Tensor()
    for model in models:
        mean, uncertainty = model.get_mean_and_std(X, predict_noiseless)
        means = torch.cat((means, mean.unsqueeze(1)), 1)
        uncertainties = torch.cat((uncertainties, uncertainty.unsqueeze(1)), 1)

    return means, uncertainties
