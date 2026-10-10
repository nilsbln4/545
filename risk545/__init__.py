"""risk545: a Python port of the course's Julia risk-management library.

Built to pass the class's functional tests (testfiles/), test group by test
group. See testfiles/Tests.xlsx / testfiles/new_test_rows.md for the test
specification this package is checked against.
"""

from .copula import (
    copula_aicc,
    copula_bic,
    fit_gaussian_copula,
    fit_t_copula,
    gaussian_copula_ll,
    simulate_gaussian_copula,
    simulate_t_copula,
    t_copula_ll,
    tail_dependence_t,
)
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
from .multivariate_t import (
    fit_multivariate_t,
    fix_correlation,
    kendall_correlation,
    mvt_loglikelihood,
    mvt_scale,
    profile_nu,
    simulate_mvt,
)
from .psd import chol_psd, higham_nearest_psd, near_psd
from .returns import return_calculate
from .risk_stats import agg_risk, expected_shortfall, value_at_risk
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
    "value_at_risk",
    "expected_shortfall",
    "agg_risk",
    "fix_correlation",
    "kendall_correlation",
    "mvt_scale",
    "mvt_loglikelihood",
    "profile_nu",
    "fit_multivariate_t",
    "simulate_mvt",
    "gaussian_copula_ll",
    "t_copula_ll",
    "fit_gaussian_copula",
    "fit_t_copula",
    "simulate_gaussian_copula",
    "simulate_t_copula",
    "tail_dependence_t",
    "copula_aicc",
    "copula_bic",
]
