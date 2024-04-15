# Welcome to ESO: Evolutionary Spectrogram Optimisation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ufuk-cakir/ESO/ci.yml?branch=main)](https://github.com/ufuk-cakir/ESO/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/ESO/badge/)](https://ESO.readthedocs.io/)
[![codecov](https://codecov.io/gh/ufuk-cakir/ESO/branch/main/graph/badge.svg)](https://codecov.io/gh/ufuk-cakir/ESO)

## Installation

The Python package `ESO` can be installed from PyPI:

```
python -m pip install ESO
```

## Development installation

If you want to contribute to the development of `ESO`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/ufuk-cakir/ESO
cd ESO
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
