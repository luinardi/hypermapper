from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hypermapper",
    version="2.2.12",
    description="HyperMapper is a multi-objective black-box optimization tool based on Bayesian Optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Luigi Nardi",
    author_email="luigi.nardi@cs.lth.se",
    url="https://github.com/luinardi/hypermapper",
    packages=["hypermapper"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "cma",
        "numpy",
        "matplotlib",
        "ply",
        "jsonschema",
        "pandas",
        "scikit-learn",
        "GPy",
        "pyDOE",
        "statsmodels",
        "ipykernel",
        "scipy",
        "threadpoolctl",
    ],
    extras_require={"plot_hvi": ["pygmo", "statsmodels"]},
    python_requires=">=3.6",
    package_data={"hypermapper": ["schema.json", "_branin_scenario.json"]},
    entry_points={
        "console_scripts": [
            "hm-compute-pareto=hypermapper._cli:_compute_pareto_cli",
            "hypermapper=hypermapper._cli:_hypermapper_cli",
            "hm-plot-pareto=hypermapper._cli:_plot_pareto_cli",
            "hm-plot-hvi=hypermapper._cli:_plot_hvi_cli",
            "hm-quickstart=hypermapper._cli:_branin_quick_start_cli",
            "hm-plot-optimization-results=hypermapper._cli:_plot_optimization_results_cli",
        ],
    },
)
