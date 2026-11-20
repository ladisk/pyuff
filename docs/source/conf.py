# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'pyuff'
copyright = '2014-2026 Primož Čermelj, Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc'
author = 'Primož Čermelj, Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc'

# The version is read from pyproject.toml, not from the installed package: Read the Docs
# installs pyuff from PyPI, whose metadata would show the last release.
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'pyproject.toml'), 'rb') as f:
    release = tomllib.load(f)['project']['version']
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

language = 'en'
master_doc = 'index'
exclude_patterns = []

# NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_member_order = 'bysource'
autodoc_typehints = 'none'
autoclass_content = 'class'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Copy code without the ">>> " and "... " prompts
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = 'furo'
html_title = f'pyuff {release}'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'source_repository': 'https://github.com/ladisk/pyuff/',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
    'navigation_with_keys': True,
}
