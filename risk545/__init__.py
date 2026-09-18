"""risk545: a Python port of the course's Julia risk-management library.

Built to pass the class's functional tests (testfiles/), test group by test
group. See testfiles/Tests.xlsx / testfiles/new_test_rows.md for the test
specification this package is checked against.
"""

from .covariance import ew_corr, ew_cov, missing_cov
from .fitted_model import (
    FittedModel,
    aicc,
    fit_general_t,
    fit_nig_mle,
    fit_nig_moments,
    fit_normal,
    fit_regression_t,
)
from .psd import chol_psd, higham_nearest_psd, near_psd
from .returns import return_calculate
from .simulate import simulate_normal, simulate_pca

__all__ = [
    "missing_cov",
    "ew_cov",
    "ew_corr",
    "near_psd",
    "higham_nearest_psd",
    "chol_psd",
    "simulate_normal",
    "simulate_pca",
    "return_calculate",
    "FittedModel",
    "aicc",
    "fit_normal",
    "fit_general_t",
    "fit_regression_t",
    "fit_nig_moments",
    "fit_nig_mle",
]
