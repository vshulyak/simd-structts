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
            #             print(endog[i,...].shape, exog.shape)
            m = UnobservedComponents(endog[i, ...], exog=exog, **kwargs)
            #             m['obs_cov', 0, 0] = 4
            #             m.ssm.initialization.set(index=None,
            #                                      initialization_type='known',
            #                                      constant=np.array([ 18.,   5., -12.,  -8.]),
            #                                      stationary_cov=np.eye(4))
            self.models += [m]

    def initialize_fixed(self):
        for m in self.models:
            m["obs_cov", 0, 0] = 0
            dims = m.ssm.transition[:, :, 0].shape[0]
            m.ssm.initialization.set(
                index=None,
                initialization_type="known",
                constant=np.zeros(dims),
                stationary_cov=np.eye(dims),
            )
        self.start_params = [1]

        if self.kwargs.get("trend", False):
            self.start_params += [1]
        if self.kwargs.get("seasonal", False):
            self.start_params += [1]
        if self.kwargs.get("freq_seasonal", False):
            self.start_params += [1] * len(m.freq_seasonal_harmonics)

        # # TODO: k_exog
        # if self.exog is not None:
        #     self.start_params += [1]

    #         self.start_params = [1,1,1,1]

    def smooth(self):
        #         start_params = [1,1]

        res = []
        for i in range(self.n_models):
            res += [
                self.models[i].smooth(
                    self.start_params,
                    transformed=True,
                    includes_fixed=False,
                    cov_type=None,
                    cov_kwds=None,
                )
            ]
        return SMResults(res)
