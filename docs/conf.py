# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "mcdapy"
copyright = "2022, Guilherme Fernandes Alves"
author = "Guilherme Fernandes Alves"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "m2r2",
]
nbsphinx_execute = "never"
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"
autoclass_content = "both"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo_link": "index",
    "github_url": "https://github.com/Gui-FernandesBR/mcdapy",
    "collapse_navigation": True,
    "show_toc_level": 3,
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}
html_theme_options = {"navbar_end": ["navbar-icon-links.html", "search-field.html"]}
html_use_modindex = True
html_copy_source = html_domain_indices = False
html_file_suffix = ".html"
htmlhelp_basename = "mcdapy"
