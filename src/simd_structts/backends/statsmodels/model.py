import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .results import SMSmootherResult


class MultiUnobservedComponents:
    def __init__(self, endog, **kwargs):
        self.exog = exog = kwargs.pop("exog", None)
        self.kwargs = kwargs
        assert endog.ndim == 3
        assert exog is None or exog.ndim in (2, 3)

        self.n_models = endog.shape[0]
        self.models = []

        for series_idx in range(self.n_models):
            exog = (
                self.exog[series_idx, ...]
                if self.exog is not None and self.exog.ndim == 3
                else self.exog
            )
            m = UnobservedComponents(endog[series_idx, ...], exog=exog, **kwargs)
            self.models += [m]

    def initialize_fixed(self, obs_cov=0, initial_state_cov=1e6):
        self.start_params = []
        for m in self.models:
            m["obs_cov", 0, 0] = obs_cov
            dims = m.ssm.transition[:, :, 0].shape[0]
            m.ssm.initialization.set(
                index=None,
                initialization_type="known",
                constant=np.zeros(dims),
                stationary_cov=np.eye(dims) * initial_state_cov,
            )
            params = [1]

            if self.kwargs.get("trend", False):
                params += [1]
            if self.kwargs.get("seasonal", False):
                params += [1]
            if self.kwargs.get("freq_seasonal", False):
                params += [1] * len(m.freq_seasonal_harmonics)

            self.start_params += [params]

    def initialize_approx_diffuse(self, obs_cov=0, initial_state_cov=1e6):
        self.start_params = []
        for m in self.models:
            m["obs_cov", 0, 0] = obs_cov
            dims = m.ssm.transition[:, :, 0].shape[0]

            m.ssm.initialization.set(
                index=None,
                initialization_type="known",
                constant=np.zeros(dims),
                stationary_cov=np.eye(dims) * initial_state_cov,
            )

            self.start_params += [m.start_params]

    def smooth(self):

        res = []
        for i in range(self.n_models):
            res += [
                self.models[i].smooth(
                    self.start_params[i],
                    transformed=True,
                    includes_fixed=False,
                    cov_type=None,
                    cov_kwds=None,
                )
            ]

        return SMSmootherResult(
            smoothed_state=np.stack([r.smoothed_state.T for r in res]),
            smoothed_state_cov=np.stack([r.smoothed_state_cov.T for r in res]),
            smoothed_forecasts=np.stack(
                [r.filter_results.smoothed_forecasts.T for r in res]
            ).squeeze(),
            smoothed_forecasts_cov=None,  # TODO
            filtered_state=np.stack([r.filtered_state.T for r in res]),
            filtered_state_cov=np.stack([r.filtered_state_cov.T for r in res]),
            predicted_state=np.stack([r.predicted_state.T for r in res]),
            predicted_state_cov=np.stack([r.predicted_state_cov.T for r in res]),
            forecast=np.stack([r.forecasts.T for r in res]),
            forecast_cov=None,  # TODO
            forecast_error_cov=np.stack([r.forecasts_error_cov.T for r in res]),
            forecast_error=np.stack([r.forecasts_error.T for r in res]),
            llf=sum([r.llf for r in res]),
            llf_obs=np.stack([r.llf_obs for r in res]),
            model=res,
        )
