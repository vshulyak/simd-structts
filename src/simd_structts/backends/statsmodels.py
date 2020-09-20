import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents


class SMPredictionResults:
    def __init__(self, forecasts):
        self.forecasts = forecasts

    @property
    def predicted_mean(self):
        return np.stack([f.predicted_mean for f in self.forecasts])

    @property
    def se_mean(self):
        return np.stack([f.se_mean for f in self.forecasts])


class SMResults:
    def __init__(self, res):
        self.res = res

    def get_forecast(self, horizon, exog=None):
        """
        TODO: check if params match to sm
        TODO: check uncertainty
        """
        return SMPredictionResults(
            [r.get_forecast(horizon, exog=exog) for r in self.res]
        )

    @property
    def filtered_state(self):
        return np.stack([r.filtered_state.T for r in self.res])

    @property
    def filtered_state_cov(self):
        return np.stack([r.filtered_state_cov.T for r in self.res])

    @property
    def smoothed_state(self):
        return np.stack([r.smoothed_state.T for r in self.res])

    @property
    def smoothed_state_cov(self):
        return np.stack([r.smoothed_state_cov.T for r in self.res])

    @property
    def smoothed_forecasts(self):
        return np.stack(
            [r.filter_results.smoothed_forecasts.T for r in self.res]
        ).squeeze()


class MultiUnobservedComponents:
    def __init__(self, endog, **kwargs):
        self.exog = exog = kwargs.pop("exog", None)
        self.kwargs = kwargs
        assert endog.ndim == 3
        assert exog is None or exog.ndim == 2

        self.n_models = endog.shape[0]
        self.models = []

        for i in range(self.n_models):
            m = UnobservedComponents(endog[i, ...], exog=exog, **kwargs)
            self.models += [m]

    def initialize_fixed(self):
        self.start_params = []
        for m in self.models:
            m["obs_cov", 0, 0] = 0
            dims = m.ssm.transition[:, :, 0].shape[0]
            m.ssm.initialization.set(
                index=None,
                initialization_type="known",
                constant=np.zeros(dims),
                stationary_cov=np.eye(dims),
            )
            params = [1]

            if self.kwargs.get("trend", False):
                params += [1]
            if self.kwargs.get("seasonal", False):
                params += [1]
            if self.kwargs.get("freq_seasonal", False):
                params += [1] * len(m.freq_seasonal_harmonics)

            self.start_params += [params]

    def initialize_approx_diffuse(self):
        self.start_params = []
        for m in self.models:
            m["obs_cov", 0, 0] = 0
            dims = m.ssm.transition[:, :, 0].shape[0]

            m.ssm.initialization.set(
                index=None,
                initialization_type="known",
                constant=np.zeros(dims),
                stationary_cov=np.eye(dims) * 1e6,
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
        return SMResults(res)
