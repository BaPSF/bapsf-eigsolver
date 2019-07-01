# bapsf-eigsolver

<!-- [![PyPI](https://img.shields.io/pypi/v/bapsflib.svg)](https://pypi.org/project/bapsflib)
[![License](https://img.shields.io/badge/License-BSD-blue.svg)](./LICENSES/LICENSE.txt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bapsflib.svg)](https://pypi.org/project/bapsflib)

[![Documentation Status](https://readthedocs.org/projects/bapsflib/badge/)](https://bapsflib.readthedocs.io/en/latest)
[![Build Status](https://img.shields.io/travis/BaPSF/bapsflib/master.svg?label=Travis%20CI)](https://travis-ci.org/BaPSF/bapsflib)
[![Build status](https://ci.appveyor.com/api/projects/status/kuia1r8iiqwiu2gg/branch/master?svg=true)](https://ci.appveyor.com/project/rocco8773/bapsflib/branch/master)
[![codecov](https://codecov.io/gh/BaPSF/bapsflib/branch/master/graph/badge.svg)](https://codecov.io/gh/BaPSF/bapsflib/branch/master)

[![h5py](https://img.shields.io/badge/powered%20by-h5py-%235e9ffa.svg)](https://www.h5py.org/)
-->
[![Documentation Status](https://readthedocs.org/projects/bapsf-eigsolver/badge/)](https://bapsf-eigsolver.readthedocs.io/en/latest)

The **bapsf-eigsolver** code was originally developed on python 2.7 and is 
currently being converted over to python 3.

**As understanding of the code improves this section will be updated.**



## Installation

Code is not registered with PyPI and has to be installed directly from GitHub
or a local copy.  See 
[installation instructions](https://bapsflib.readthedocs.io/en/latest/installation.html) 
from **bapsflib** documentation as a reference.

<!--
**bapsflib** is registered with [PyPI](https://pypi.org/) and can be 
installed with `pip` via

`pip install bapsflib`

To install from source look to installation instructions in 
documentation, 
[here](https://bapsflib.readthedocs.io/en/latest/installation.html).
-->

## Documentation

The documentation is hosted on Read the Docs at 
https://bapsf-eigsolver.readthedocs.io/en/latest. [**NOTE:** Since this repo
is a private repo, the Read the Docs version is not updated.]

To build a local copy of documentation navigate to the `./docs` folder and
execute 

`make clean`

to clean past builds and 

`make html`

to build a new html version of the documentation.  Go to, 
`./docs/_build/html/index.html` to open the freshly built documentation.

## How to run test script `drift.py`

The test script in located at `bapsf_eigsolver/example/drift.py`.

* To run in a python 3 session

   ``` python
   with open('<path to script>/drift.py) as f:
       exec(f.read())
   ```

* To run in an IPython session `%run <path to script>/drift.py`.

* To run in terminal `python3 -m bapsf_eigsolver.example.drift`.