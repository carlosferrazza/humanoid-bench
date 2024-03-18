import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "Embodied"
copyright = "2021, Danijar Hafner"
author = "Danijar Hafner"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autodocsumm",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = ["numpy"]
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "member-order": "bysource",
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
}

html_theme = "sphinx_rtd_theme"
