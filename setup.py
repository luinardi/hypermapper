from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hypermapper",
    version="3.0.0",
    description="HyperMapper is a multi-objective black-box optimization tool based on Bayesian Optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Luigi Nardi",
    author_email="luigi.nardi@cs.lth.se",
    url="https://github.com/hypermapper/hypermapper",
    packages=find_packages(include=["hypermapper"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.24.4",
        "setuptools>=68.0.0",
        "torch>=2.0.1",
        "GPy>=1.10.0",
        "scipy>=1.10.1",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "jsonschema>=4.19.0",
        "numexpr>=2.8.5",
        "networkx>=3.1",
        "graphviz>=0.20.1",
        "botorch>=0.8.5",
        "gpytorch>=1.10",
    ],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["hypermapper = hypermapper.run:main"]},
)
