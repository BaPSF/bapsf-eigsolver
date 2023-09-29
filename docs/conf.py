#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'bapsf-eigsolver'
copyright = '2019, Erik T. Everson, Phil Travis, Abhishek Shetty'
author = 'Erik T. Everson, Phil Travis, Abhishek Shetty'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
]

numfig = True  # enable figure and tube numbering
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = ['.rst', ]

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce
# nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- My Added Extras ---------------------------------------------------

# A list of prefixes that are ignored for sorting the Python module
# index (e.g., if this is set to ['foo.'], then foo.bar is shown under
# B, not F).
modindex_common_prefix = ['bapsf_eigsolver.']

# prevents files that match these patterns from being included in source
# files
# - prevents file file.inc.rst from being loaded twice: once when
#   included as a source file and second when it's inserted into another
#   .rst file with .. include
#
exclude_patterns.extend([
    '**.inc.rst'
])

# add a pycode role for inline markup e.g. :pycode:`'mycode'`
rst_prolog = """
.. role:: pycode(code)
   :language: python3

.. role:: red
.. role:: green
.. role:: blue

.. role:: ibf
    :class: ibf

.. role:: textit
    :class: textit

.. role:: textbf
    :class: textbf
"""


def setup(app):
    # add bapsf CSS overrides
    app.add_css_file("bapsf_overrides.css", priority=600)
    return app
