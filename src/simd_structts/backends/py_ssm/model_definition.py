from dataclasses import dataclass

import numpy as np

from .filter import FILTER_CONVENTIONAL
from .filter import INVERT_UNIVARIATE
from .filter import MEMORY_STORE_ALL
from .filter import SOLVE_CHOLESKY
from .filter import STABILITY_FORCE_SYMMETRY


@dataclass
class ModelDefinition:

    obs: np.ndarray

    # SS def
    selection: np.ndarray
    state_cov: np.ndarray
    design: np.ndarray
    obs_intercept: np.ndarray
    obs_cov: np.ndarray
    transition: np.ndarray
    state_intercept: np.ndarray
    time_invariant: bool
    k_endog: int

    # dynamic
    nobs: int
    k_states: int

    dtype = np.float64

    # Kalman filter properties
    filter_method = FILTER_CONVENTIONAL
    inversion_method = INVERT_UNIVARIATE | SOLVE_CHOLESKY
    stability_method = STABILITY_FORCE_SYMMETRY
    conserve_memory = MEMORY_STORE_ALL
    tolerance = 0
