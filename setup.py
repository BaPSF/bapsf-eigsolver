#!/usr/bin/env python3
#
# This file is part of the bapsf-egisolver package.
#
# http://plasma.physics.ucla.edu/
#
# Copyright 2019 Erik T. Everson and contributors
#
import codecs
import os
import re

from setuptools import setup, find_packages

# find here
here = os.path.abspath(os.path.dirname(__file__))

# define CLASSIFIERS
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# ---- Define helpers for version-ing                               ----
# - following 'Single-sourcing the package version' from 'Python
#   Packaging User Guide'
#   https://packaging.python.org/guides/single-sourcing-package-version/
#
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# ---- Define long description                                      ----
with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

# ---- Perform setup                                                ----
setup(
    name='bapsf-eigsolver',
    version=find_version("bapsf_eigsolver", "__init__.py"),
    description='Eigenvalue solver for modes in the LaPD at UCLA.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    scripts=[],
    setup_requires=['numpy>=1.12',
                    'sympy>=1.1.1',
                    'scipy>=1.0.0',
                    'matplotlib>=2.0'],
    install_requires=['numpy>=1.12',
                      'sympy>=1.1.1',
                      'scipy>=1.0.0',
                      'matplotlib>=2.0'],
    python_requires='>=3.5',
    author='Erik T. Everson',
    author_email='eteveson@gmail.com',
    license='',
    url='https://github.com/BaPSF/bapsf-eigsolver',
    keywords=['bapsf', 'lapd', 'physics', 'plasma', 'science'],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    package_urls={
        "BaPSF": "http://plasma.physics.ucla.edu/",
        "Documentation": "https://bapsf-eigsolver.readthedocs.io/en/latest/",
        "GitHub": "https://github.com/BaPSF/bapsf-eigsolver",
    }
)
