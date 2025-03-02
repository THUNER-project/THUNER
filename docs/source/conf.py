# Configuration file for the Sphinx documentation builder.
# First add the project root to the path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information

project = "THUNER"
copyright = "2025, THUNER-project"
author = "Ewan Short"

release = "0.0.16"
version = "0.0.16"

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autodoc_pydantic",
]

autodoc_default_options = {"exclude-members": "model_post_init"}

autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_model_summary_list_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_model_show_config_summary = False

autodoc_mock_imports = [
    "numba",
    "scipy",
    "scikit-image",
    "cartopy",
    "typing_extensions",
    "codecov",
    "netcdf4",
    "h5netcdf",
    "requests",
    "tqdm",
    "cdsapi",
    "xesmf",
    "opencv",
    "nco",
    "pytables",
    "pydantic",
    "zarr",
    "windrose",
    "pydot",
    "metpy",
    "graphviz",
    "pygraphviz",
    "nbconvert",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
