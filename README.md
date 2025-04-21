<p align="center">
  <img src="https://cakir-ufuk.de/assets/images/eso%20logo.png" />
</p>

# Welcome to eso: Evolutionary Spectrogram Optimisation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ufuk-cakir/eso/ci.yml?branch=main)](https://github.com/ufuk-cakir/eso/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/eso/badge/)](https://eso.readthedocs.io/)
[![codecov](https://codecov.io/gh/ufuk-cakir/eso/branch/main/graph/badge.svg)](https://codecov.io/gh/ufuk-cakir/eso)


# Note
> :warning: **The main branch is currently under construction. We will add proper unit testing soon. Check the [develop branch](https://github.com/ufuk-cakir/eso/tree/develop) for an experimental API.**


## Installation

The Python package `eso` can be installed from PyPI:

```
python -m pip install eso
```

## Development installation

If you want to contribute to the development of `eso`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/ufuk-cakir/eso
cd eso
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
