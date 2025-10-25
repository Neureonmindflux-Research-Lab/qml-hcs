import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath('..'))

project = "qmlhc Minimal Core Demo"
author = "Neureonmindflux Research Lab"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",            
    "sphinx.ext.napoleon",           
    "sphinx.ext.viewcode",         
    "sphinx_autodoc_typehints",     
    "myst_parser",                   
    "sphinx.ext.mathjax",   
    "sphinxcontrib.programoutput",        
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/qml-hcs-icon.svg"


# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_param = True
napoleon_use_rtype = True


autodoc_mock_imports = [
    "qiskit",
    "pennylane",
    "torch",
    "jax",
    "jaxlib",
]
