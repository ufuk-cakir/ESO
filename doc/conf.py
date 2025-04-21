# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(".."))

print("sys.path", sys.path)
# -- Project information -----------------------------------------------------

project = "eso"
copyright = "2024, Ufuk Çakır"
author = "Ufuk Çakır"
nbsphinx_execute = "never"
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx_link",
    "sphinx_mdinclude",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "pydata-sphinx-theme"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# conf.py

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Make Sphinx warn on everything it can’t resolve
nitpicky = True
keep_warnings = True
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
