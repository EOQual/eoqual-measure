# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
from eoqual.measure import __version__ as _version  # noqa: E402

project = 'eoqual-measure'
copyright = '2026, Olivier Amram'
author = 'Olivier Amram'
release = _version
version = _version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'autoapi.extension']


templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/img/eoqual-logo.png'
html_favicon = '_static/img/eoqual-favicon.png'

html_css_files = [
    'css/custom.css',
]

inheritance_graph_attrs = dict(rankdir="TB", size='""')

autoapi_dirs = ['../../src']
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autoapi_keep_files = True
autodoc_typehints = "signature"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "imported-members",
]
