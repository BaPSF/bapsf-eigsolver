"""Root module for the `bapsf_eigsolver` package."""
__all__ = ["eigsolver", "__version__"]

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

if sys.version_info < (3, 7):  # coverage: ignore
    raise ImportError("bapsf_eigsolver does not support Python < 3.7")

import pkg_resources

from bapsf_eigsolver import eigsolver

# define version
try:
    # this places a runtime dependency on setuptools
    #
    # note: if there's any distribution metadata in your source files, then this
    #       will find a version based on those files.  Keep distribution metadata
    #       out of your repository unless you've intentionally installed the package
    #       as editable (e.g. `pip install -e {bapsf_eigsolver_directory_root}`),
    #       but then __version__ will not be updated with each commit, it is
    #       frozen to the version at time of install.
    #
    #: bapsf_eigsolver version string
    __version__ = pkg_resources.get_distribution("bapsf_eigsolver").version
except pkg_resources.DistributionNotFound:
    # package is not installed
    fallback_version = "unknown"
    try:
        # code most likely being used from source
        # if setuptools_scm is installed then generate a version
        from setuptools_scm import get_version

        __version__ = get_version(
            root="..", relative_to=__file__, fallback_version=fallback_version
        )
        del get_version
        warn_add = "setuptools_scm failed to detect the version"
    except ModuleNotFoundError:
        # setuptools_scm is not installed
        __version__ = fallback_version
        warn_add = "setuptools_scm is not installed"

    if __version__ == fallback_version:
        from warnings import warn

        warn(
            f"bapsf_eigsolver.__version__ not generated (set to 'unknown'), "
            f"bapsf_eigsolver is not an installed package and {warn_add}.",
            RuntimeWarning,
        )

        del warn
    del fallback_version, warn_add
