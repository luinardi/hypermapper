import os
import hypermapper  # noqa
from hypermapper.util.file import extend_with_default
from typing import Callable, Optional, List, Dict
import argparse
import json
import jsonschema
from pkg_resources import resource_stream
import torch

from botorch.test_functions.synthetic import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DropWave,
    DixonPrice,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
    ThreeHumpCamel,
)


testing_directory = os.getcwd()

if not os.path.isdir(os.path.join(f"{testing_directory}", "logs")):
    os.mkdir(os.path.join(f"{testing_directory}", "logs"))
if not os.path.isdir(os.path.join(f"{testing_directory}", "outputfiles")):
    os.mkdir(os.path.join(f"{testing_directory}", "outputfiles"))
if not os.path.isdir(os.path.join(f"{testing_directory}", "scenarios")):
    os.mkdir(os.path.join(f"{testing_directory}", "scenarios"))


class BotorchBenchmark:
    def __init__(self, name: str, dim: int = 2, noise_std: float = 0.0):
        self.name = name
        self.dim = dim
        self.noise_std = noise_std

        if name == "ackley":
            self.btbench = Ackley(dim=dim, noise_std=noise_std)
        if name == "branin":
            self.btbench = Branin(noise_std=noise_std)
        elif name == "beale":
            self.btbench = Beale(noise_std=noise_std)
        elif name == "bukin":
            self.btbench = Bukin(noise_std=noise_std)
        elif name == "cosine8":
            self.btbench = Cosine8(noise_std=noise_std)
        elif name == "dropwave":
            self.btbench = DropWave(noise_std=noise_std)
        elif name == "dixonprice":
            self.btbench = DixonPrice(dim=dim, noise_std=noise_std)
        elif self.name == "eggholder":
            self.btbench = EggHolder(noise_std=noise_std)
        elif self.name == "griewank":
            self.btbench = Griewank(dim=dim, noise_std=noise_std)
        elif self.name == "hartmann":
            self.btbench = Hartmann(dim=dim, noise_std=noise_std)
        elif self.name == "holdertable":
            self.btbench = HolderTable(noise_std=noise_std)
        elif self.name == "levy":
            self.btbench = Levy(dim=dim, noise_std=noise_std)
        elif self.name == "michalewicz":
            self.btbench = Michalewicz(dim=dim, noise_std=noise_std)
        elif self.name == "powell":
            self.btbench = Powell(dim=dim, noise_std=noise_std)
        elif self.name == "rastrigin":
            self.btbench = Rastrigin(dim=dim, noise_std=noise_std)
        elif self.name == "rosenbrock":
            self.btbench = Rosenbrock(dim=dim, noise_std=noise_std)
        elif self.name == "shekel":
            self.btbench = Shekel(m=10, noise_std=noise_std)
        elif self.name == "sixhumpcamel":
            self.btbench = SixHumpCamel(noise_std=noise_std)
        elif self.name == "styblinskitang":
            self.btbench = StyblinskiTang(dim=dim, noise_std=noise_std)
        elif self.name == "threehumpcamel":
            self.btbench = ThreeHumpCamel(noise_std=noise_std)
        else:
            raise Exception(
                f"Benchmark name {self.name} not available. Please choose from the following: "
                + "ackley, beale, branin, bukin, cosine8, dropwave, dixonprice, eggholder, griewank, "
                + "hartmann, holdertable, levy, michalewicz, powell, rastrigin, rosenbrock, shekel, "
                + "sixhumpcamel,styblinskitang, threehumpcamel"
            )
        self.param_names = [f"x{i+1}" for i in range(self.btbench.dim)]

    def get_parameters(self):
        param_dict = {}
        for i, bound in enumerate(self.btbench.bounds.T):
            param_dict[f"x{i + 1}"] = {
                "parameter_type": "real",
                "values": bound.numpy().tolist(),
                "parameter_default": ((bound[0] + bound[1]) / 2).item(),
            }
        return param_dict

    def __call__(self, x):
        if isinstance(x, dict):
            x = torch.tensor([x[f"x{i+1}"] for i in range(self.btbench.dim)])
        return self.btbench.forward(x)

    def get_name(self):
        return self.name


def run_performance_test(
    benchmark: BotorchBenchmark,
    doe_samples: int = 10,
    iterations: int = 50,
    repetitions: int = 1,
    additional_settings: Dict = None,
    run_tag: str = "notag",
):
    if additional_settings is None:
        additional_settings = {}

    schema = json.load(resource_stream("hypermapper", "schema.json"))
    settings = {
        "application_name": benchmark.name,
        "optimization_iterations": iterations,
        "design_of_experiment": {
            "doe_type": "random sampling",
            "number_of_samples": doe_samples,
        },
        "optimization_objectives": ["Value"],
        "input_parameters": benchmark.get_parameters(),
        "hypermapper_mode": {"mode": "default"},
        "optimization_method": "bayesian_optimization",
    }
    extend_with_default(jsonschema.Draft4Validator)(schema).validate(settings)
    settings.update(additional_settings)

    if not os.path.isdir(os.path.join(f"{testing_directory}", "logs", run_tag)):
        os.mkdir(os.path.join(f"{testing_directory}", "logs", run_tag))
    if not os.path.isdir(os.path.join(f"{testing_directory}", "outputfiles", run_tag)):
        os.mkdir(os.path.join(f"{testing_directory}", "outputfiles", run_tag))
    if not os.path.isdir(os.path.join(f"{testing_directory}", "scenarios", run_tag)):
        os.mkdir(os.path.join(f"{testing_directory}", "scenarios", run_tag))

    for rep in range(repetitions):
        print("Running:", benchmark.name, "Tag:", run_tag, "Repetition:", rep)
        settings["log_file"] = os.path.join(
            f"{testing_directory}", "logs", run_tag, f"log_{benchmark.name}_{rep}.log"
        )
        settings["output_data_file"] = os.path.join(
            f"{testing_directory}",
            "outputfiles",
            run_tag,
            f"out_{benchmark.name}_{rep}.csv",
        )
        settings["run_directory"] = testing_directory
        json.dump(
            settings,
            open(
                os.path.join(
                    f"{testing_directory}", "scenarios", run_tag, f"scenario_{rep}.json"
                ),
                "w",
            ),
            indent=4,
        )
        hypermapper.optimize(settings, benchmark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark")

    parser.add_argument("benchmark", type=str, help="Name of the benchmark to run")
    parser.add_argument(
        "--dim", type=int, help="Dimension of the benchmark", default=-1, dest="dim"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        help="Noise std of the benchmark",
        default=0,
        dest="noise_std",
    )
    parser.add_argument(
        "--doe_samples", type=int, help="DoE samples", dest="doe_samples"
    )
    parser.add_argument("--iterations", type=int, help="Iterations", dest="iterations")
    parser.add_argument(
        "--repetitions", type=int, help="Repetitions", default=1, dest="repetitions"
    )
    parser.add_argument(
        "--tag", type=str, help="Tag for the run", default="notag", dest="tag"
    )
    parser.add_argument("--settings", "-s", nargs="*", default=[])

    args = parser.parse_args()

    benchmark_name = args.benchmark
    dim = args.dim
    noise_std = args.noise_std
    doe_samples = args.doe_samples
    iterations = args.iterations
    repetitions = args.repetitions
    run_tag = args.tag
    settings = args.settings

    assert (
        benchmark_name is not None
    ), "Please specify the benchmark name as the first positional argument"
    assert (
        doe_samples is not None
    ), "Please specify the number of DoE samples using --doe_samples"
    assert (
        iterations is not None
    ), "Please specify the number of iterations using --iterations"

    settings_dict = {}
    for s in settings:
        key, value = s.split(":")
        try:
            v = float(value)
        except ValueError:
            v = value
        if isinstance(v, float):
            if v % 1 == 0:
                v = int(v)
        settings_dict[key] = v

    benchmark = BotorchBenchmark(name=benchmark_name, dim=dim, noise_std=noise_std)
    run_performance_test(
        benchmark,
        doe_samples=doe_samples,
        iterations=iterations,
        repetitions=repetitions,
        additional_settings=settings_dict,
        run_tag=run_tag,
    )
