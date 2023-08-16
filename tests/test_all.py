import os
import sys

import torch
from matplotlib import pyplot as plt

import hypermapper  # noqa
from aux.functions import *
from aux.test_cli import branin4_cli
from typing import Callable, Optional, List, Dict
from hypermapper.util.file import read_settings_file

testing_directory = os.path.dirname(__file__)

if not os.path.isdir(os.path.join(f"{testing_directory}", "logs")):
    os.mkdir(os.path.join(f"{testing_directory}", "logs"))
if not os.path.isdir(os.path.join(f"{testing_directory}", "outputfiles")):
    os.mkdir(os.path.join(f"{testing_directory}", "outputfiles"))

def runBenchmark(scenario: str, function: Callable, output_file: Optional[str] = None):
    settings_file = os.path.join(f"{testing_directory}", "aux", f"{scenario}.json")
    settings = read_settings_file(settings_file)
    settings['log_file'] = os.path.join(f"{testing_directory}", "logs", f"{scenario.split('.')[0]}.log")
    settings['resume_optimization_file'] = os.path.join(f"{testing_directory}", "aux", f"{settings['resume_optimization_file']}")
    if output_file:
        settings["output_data_file"] = os.path.join(f"{testing_directory}", "outputfiles", f"{output_file}.csv")
    else:
        settings["output_data_file"] = os.path.join(f"{testing_directory}", "outputfiles", f"{scenario.split('.')[0]}.csv")
    hypermapper.optimize(settings, function)


def run_performance_test(
        setting_files: List[str],
        functions: List[Callable],
        benchmark_names: List[str],
        additional_settings: List[Dict],
        names: List[str],
        n_runs: int
):
    ITERATIONS = 200
    DOE = 10

    for filename, function, bname in zip(setting_files, functions, benchmark_names):
        for setting, name in zip(additional_settings, names):
            settings_file = os.path.join(f"{testing_directory}", "aux", f"{filename}.json")
            settings = read_settings_file(settings_file)
            settings.update(setting)
            settings["optimization_iterations"] = ITERATIONS
            settings["design_of_experiment"]["number_of_samples"] = DOE
            for j in range(n_runs):
                print("Running", bname, name, j)
                settings['log_file'] = os.path.join(f"{testing_directory}", "logs", f"log_{bname}_{name}_{j}.log")
                settings["output_data_file"] = os.path.join(f"{testing_directory}", "outputfiles", f"out_{bname}_{name}_{j}.csv")
                hypermapper.optimize(settings, function)


def plot_test(
        setting_files: List[str],
        functions: List[Callable],
        benchmark_names: List[str],
        additional_settings: List[Dict],
        names: List[str],
        n_runs: int,
):
    for filename, function, bname in zip(setting_files, functions, benchmark_names):
        fig, ax = plt.subplots()
        for setting, name in zip(additional_settings, names):
            y = []
            for run in range(n_runs):
                data = np.genfromtxt(os.path.join(f"{testing_directory}", "outputfiles", f"out_{bname}_{name}_{run}.csv"), skip_header=1, delimiter=',', dtype=np.float64)
                y.append(data[:, -2])
            y = np.log10(np.array(y))
            y = torch.cummin(torch.Tensor(y), dim=1).values.numpy()
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            ax.plot(y_mean, label=f"{bname}_{name}")
            # ax.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax.legend()
        # ax.set_yscale('log')
    plt.show()

    for filename, function, bname in zip(setting_files, functions, benchmark_names):
        fig, ax = plt.subplots()
        for setting, name in zip(additional_settings, names):
            fss = []
            mus = []
            sigmas = []
            for run in range(n_runs):
                fs = []
                mu = []
                sigma = []
                with open(os.path.join(f"{testing_directory}", "logs", f"log_{bname}_{name}_{run}.log")) as f:
                    yes = False
                    for line in f:
                        if "Multi-start LS time" in line:
                            yes = True
                        if "f*" in line and yes:
                            fs.append(float(line.split("f*:")[-1].split("mu")[0]))
                            mu.append(float(line.split("mu:")[-1].split("sigma")[0]))
                            sigma.append(float(line.split("sigma:")[-1].split("feasibility")[0]))
                            yes = False
                fss.append(fs)
                mus.append(mu)
                sigmas.append(sigma)
            fss = np.array(fss)
            mus = np.array(mus)
            sigmas = np.array(sigmas)
            print(fss.shape, mus.shape, sigmas.shape)
            fss_mean = np.mean(fss, axis=0)
            mus_mean = np.mean(mus, axis=0)
            sigmas_mean = np.mean(sigmas, axis=0)
            diff_mean = np.mean(mus-fss, axis=0)
            sigdiff_mean = np.mean(sigmas/(mus-fss), axis=0)
            # smooth_mean = [np.mean(sigdiff_mean[i:i+5]) for i in range(len(sigdiff_mean)-5)]
            ax.plot(sigmas_mean, label=f"{bname}_{name}")
            # ax.plot(smooth_mean, label=f"{bname}_{name}")
            ax.set_title(f"estimated standard deviation")
            # ax.set_title(f"smoothed(5) penalized estimated standard deviation")

            # ax.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax.legend()
        # ax.set_yscale('log')
    plt.show()


def performance_test():
    setting_files = [
        "branin4_scenario_gp",
        "branin4_scenario_integer",
        "branin4_scenario_feas",
        "branin4_scenario_discrete",
        "rs_cot_1024_scenario",
    ]
    functions = [
        branin4_function,
        branin4_function,
        branin4_function_feas,
        branin4_function,
        rs_cot_1024,
    ]
    benchmark_names = [
        "branin4_gp",
        "branin4_integer",
        "branin4_feas",
        "branin4_discrete",
        "rs_cot_1024",
    ]
    additional_settings = [
        {"GP_model": "gpy"},
        {"GP_model": "gpytorch"},
        {"GP_model": "botorch"},
    ]
    names = ["GPy", "GPyTorch", "BoTorch"]
    run_performance_test(setting_files, functions, benchmark_names, additional_settings, names, 1)

def rf_test(plot=False):
    setting_files = [
        "branin4_scenario_rf",
    ]
    functions = [
        branin4_function,
    ]
    benchmark_names = [
        "branin4_rf",
    ]
    additional_settings = [
        {"models": {"model": "random_forest", "number_of_trees": 200, "max_features": 0.5, "min_samples_split": 2, "bootstrap": "True", "use_all_data_to_fit_mean": "False", "use_all_data_to_fit_variance": "False"}},
        {"models": {"model": "random_forest", "number_of_trees": 200, "max_features": 0.5, "min_samples_split": 2, "bootstrap": "True", "use_all_data_to_fit_mean": "True", "use_all_data_to_fit_variance": "False"}},
        {"models": {"model": "random_forest", "number_of_trees": 200, "max_features": 0.5, "min_samples_split": 2, "bootstrap": "True", "use_all_data_to_fit_mean": "True", "use_all_data_to_fit_variance": "True"}},
        {"models": {"model": "random_forest", "number_of_trees": 200, "max_features": 1.0, "min_samples_split": 2, "bootstrap": "True", "use_all_data_to_fit_mean": "False", "use_all_data_to_fit_variance": "False"}},
        # {"models": {"model": "random_forest", "number_of_trees": 2000, "max_features": 1.0, "min_samples_split": 2, "bootstrap": "True", "use_all_data_to_fit_mean": "False", "use_all_data_to_fit_variance": "False"}},
    ]
    #names = ["base", "max_features=1.0", "max_features=0.2", "min_samples_split=5", "n_estimators=10", "bootstrap=False"]
    names = ["new_base", "mean+", "var+", "new_base_max1"]
    # names = ["new_base_max1", "new_base_max1_2000"]
    if plot:
        plot_test(setting_files, functions, benchmark_names, additional_settings, names, 5)
    else:
        run_performance_test(setting_files, functions, benchmark_names, additional_settings, names, 10)


def crash_test():
    # GP
    print("GP")
    runBenchmark("branin4_scenario_gp", branin4_function)
    runBenchmark("branin4_scenario_gpy", branin4_function)
    runBenchmark("branin4_scenario_gpytorch", branin4_function)

    # RF
    print("RF")
    runBenchmark("branin4_scenario_rf", branin4_function)

    # Integer
    print("Integer")
    runBenchmark("branin4_scenario_integer", branin4_function)

    # Resume
    print("Resume")
    runBenchmark("branin4_scenario_resume", branin4_function)

    # Constraints
    print("Constraints")
    runBenchmark("branin4_scenario_feas", branin4_function_feas)

    # DISCRETE
    print("Discrete")
    runBenchmark("branin4_scenario_discrete", branin4_function)
    runBenchmark("rs_cot_1024_scenario", rs_cot_1024)

    # PIBO
    print("PIBO")
    runBenchmark("branin4_scenario_pibo", branin4_function)
    runBenchmark("branin4_scenario_pibo_real", branin4_function)

    # PERM
    print("PERM")
    runBenchmark("perm", perm)

    # RS
    settings_file = os.path.join(f"{testing_directory}", "aux", "branin4_scenario_gp.json")
    settings = read_settings_file(settings_file)
    settings['log_file'] = os.path.join(f"{testing_directory}", "logs", "branin4_scenario_gp_rs.log")
    settings['output_data_file'] = os.path.join(f"{testing_directory}", "outputfiles", "branin4_scenario_gp_rs.csv")
    settings["design_of_experiment"]["number_of_samples"] = 30
    settings["optimization_iterations"] = 0
    hypermapper.optimize(settings, branin4_function)

    # CLI
    branin4_cli(testing_directory)


if __name__ == "__main__":
    print("Running all tests.")
    # performance_test()
    # rf_test(plot=True)
    crash_test()