# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mcdapy"
copyright = "2023, Guilherme Fernandes Alves"
author = "Guilherme Fernandes Alves"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    "m2r2",
    "nbsphinx",
]

# templates_path = ["_templates"]
# exclude_patterns = ["_build"]
napoleon_numpy_docstring = True
# numpydoc_show_class_members = False
nbsphinx_execute = "never"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["static"]
html_use_modindex = True
html_file_suffix = ".html"
htmlhelp_basename = "mcdapy"
