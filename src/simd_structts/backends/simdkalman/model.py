import numpy as np
from statsmodels.tsa.statespace.tools import companion_matrix

from .kalman_filter import EKalmanFilter


class SIMDKalmanPredictionResults:
    def __init__(self, predicted):
        self.predicted = predicted

    @property
    def predicted_mean(self):
        return self.predicted.observations.mean

    @property
    def se_mean(self):
        return self.predicted.observations.cov ** (1 / 2)


class SIMDKalmanFilterResults:
    def __init__(
        self, endog, kf, exog=None, initial_value=None, initial_covariance=None
    ):
        self.endog = endog
        self.kf = kf
        self.exog = exog
        self.initial_value = initial_value
        self.initial_covariance = initial_covariance

    def _compute(self, horizon=0, exog=None):
        # TODO: optimize
        if horizon > 0 and exog is not None:

            n_obs, _, k_comp = self.kf.observation_model.shape

            n_exog, k_exog = exog.shape

            # take last row of the design matrix and clone it n_exog times
            design = np.repeat(self.kf.observation_model[-2:-1], n_exog, axis=0)
            design[:, 0, k_comp - k_exog : k_comp] = exog

            observation_model = np.vstack([self.kf.observation_model, design])

            kf = EKalmanFilter(
                state_transition=self.kf.state_transition,
                observation_model=observation_model,
                process_noise=self.kf.process_noise,
                observation_noise=self.kf.observation_noise,
            )
        else:
            kf = self.kf

        return kf.compute(
            self.endog,
            horizon,
            filtered=True,
            smoothed=True,
            initial_value=self.initial_value,
            initial_covariance=self.initial_covariance,
        )

    def get_forecast(self, horizon, exog=None):
        """
        TODO: check if params match to sm
        TODO: check uncertainty
        """
        return SIMDKalmanPredictionResults(
            self._compute(horizon=horizon, exog=exog).predicted
        )

    @property
    def filtered_state(self):
        return self._compute().filtered.states.mean

    @property
    def filtered_state_cov(self):
        return self._compute().filtered.states.cov

    @property
    def smoothed_state(self):
        return self._compute().smoothed.states.mean

    @property
    def smoothed_state_cov(self):
        return self._compute().smoothed.states.cov

    @property
    def smoothed_forecasts(self):
        return self._compute().smoothed.observations.mean


class SIMDStructTS:
    def __init__(
        self,
        endog,
        level=False,
        trend=False,
        seasonal=None,
        freq_seasonal=None,
        cycle=False,
        autoregressive=None,
        exog=None,
        irregular=False,
        stochastic_level=False,
        stochastic_trend=False,
        stochastic_seasonal=True,
        stochastic_freq_seasonal=None,
        stochastic_cycle=False,
        damped_cycle=False,
        cycle_period_bounds=None,
        mle_regression=True,
        use_exact_diffuse=False,
    ):
        self.endog = endog

        # cycle NA
        assert cycle is False
        assert stochastic_cycle is False
        assert damped_cycle is False
        assert cycle_period_bounds is None
        # autoregressive NA
        assert autoregressive is None
        # irregular NA
        assert irregular is False

        assert exog is None or exog.ndim == 2
        assert (
            mle_regression is False
        ), "MLE is not supported for estimating params currently"

        self.mle_regression = mle_regression

        # Model options
        self.level = level
        self.trend = trend
        self.seasonal_periods = seasonal if seasonal is not None else 0
        self.seasonal = self.seasonal_periods > 0
        if freq_seasonal:
            self.freq_seasonal_periods = [d["period"] for d in freq_seasonal]
            self.freq_seasonal_harmonics = [
                d.get("harmonics", int(np.floor(d["period"] / 2)))
                for d in freq_seasonal
            ]
        else:
            self.freq_seasonal_periods = []
            self.freq_seasonal_harmonics = []
        self.freq_seasonal = any(x > 0 for x in self.freq_seasonal_periods)
        self.cycle = cycle
        self.ar_order = autoregressive if autoregressive is not None else 0
        self.autoregressive = self.ar_order > 0
        self.irregular = irregular

        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        if stochastic_freq_seasonal is None:
            self.stochastic_freq_seasonal = [True] * len(self.freq_seasonal_periods)
        else:
            if len(stochastic_freq_seasonal) != len(freq_seasonal):
                raise ValueError(
                    "Length of stochastic_freq_seasonal must equal length"
                    " of freq_seasonal: {!r} vs {!r}".format(
                        len(stochastic_freq_seasonal), len(freq_seasonal)
                    )
                )
            self.stochastic_freq_seasonal = stochastic_freq_seasonal
        self.stochastic_cycle = stochastic_cycle

        self.k_series = endog.shape[0]
        self.nobs = endog.shape[1]

        # Exogenous component
        self.k_exog = exog.shape[1] if exog is not None else 0
        self.exog = exog

        self.regression = self.k_exog > 0

        # Model parameters
        self._k_seasonal_states = (self.seasonal_periods - 1) * self.seasonal
        self._k_freq_seas_states = (
            sum(2 * h for h in self.freq_seasonal_harmonics) * self.freq_seasonal
        )
        self._k_cycle_states = self.cycle * 2

        self.k_states = k_states = (
            self.level
            + self.trend
            + self._k_seasonal_states
            + self._k_freq_seas_states
            + self._k_cycle_states
            + self.ar_order
            + (not self.mle_regression) * self.k_exog
        )

        # initial SSM matrices
        self.transition = np.zeros((k_states, k_states))
        self.design = np.zeros((1, k_states))
        self.state_cov = np.zeros((self.k_series, k_states, k_states))
        self.obs_cov = np.array([[0.0]])

        self.initial_value = np.ones(k_states)
        self.initial_covariance = np.eye(4)

        self.setup()

    def initialize_fixed(self):

        for series_idx in range(self.k_series):

            offset = 0

            # level
            self.state_cov[series_idx, offset, offset] = 1

            # trend
            if self.trend:
                offset += 1
                self.state_cov[series_idx, offset, offset] = 1

            # seasonal
            if self.seasonal:
                offset += 1
                self.state_cov[series_idx, offset, offset] = 1

                # account for added seasonal components
                offset += self._k_seasonal_states - 1

            # freq_seasonal
            for _ in range(self._k_freq_seas_states):
                offset += 1
                self.state_cov[series_idx, offset, offset] = 1

        self.obs_cov[0, 0] = 0
        self.initial_value = np.zeros(self.k_states)
        self.initial_covariance = np.eye(self.k_states)

    def initialize_approx_diffuse(self):

        #         sigma_epsilon = 2.0 # affects the measurement error
        #         sigma_xi = 1.0 # affects the local level
        #         sigma_omega = 1.0 # affects the seasonality
        #         self.state_cov[0,0] = sigma_xi ** 2
        #         self.state_cov[1,1] = sigma_omega ** 2
        #         self.obs_cov[0,0] = sigma_epsilon ** 2

        from statsmodels.tsa.filters.hp_filter import hpfilter

        # Eliminate missing data to estimate starting parameters
        endog = self.endog
        exog = self.exog
        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]

        for series_idx in range(self.k_series):

            # Level / trend variances
            # (Use the HP filter to get initial estimates of variances)
            _start_params = {}

            resid, trend1 = hpfilter(endog[series_idx, :])

            if self.stochastic_trend:
                cycle2, trend2 = hpfilter(trend1)
                _start_params["trend_var"] = np.std(trend2) ** 2
                if self.stochastic_level:
                    _start_params["level_var"] = np.std(cycle2) ** 2
            elif self.stochastic_level:
                _start_params["level_var"] = np.std(trend1) ** 2

            # The variance of the residual term can be used for all variances,
            # just to get something in the right order of magnitude.
            var_resid = np.var(resid)

            # Seasonal
            if self.stochastic_seasonal:
                _start_params["seasonal_var"] = var_resid

            # Frequency domain seasonal
            if self.stochastic_freq_seasonal:
                _start_params["freq_seasonal_var"] = var_resid

            offset = 0

            # level
            self.state_cov[series_idx, offset, offset] = _start_params["level_var"]

            # trend
            if self.trend:
                offset += 1
                self.state_cov[series_idx, offset, offset] = _start_params["trend_var"]

            # seasonal
            if self.seasonal:
                offset += 1
                self.state_cov[series_idx, offset, offset] = _start_params[
                    "seasonal_var"
                ]

                # account for added seasonal components
                offset += self._k_seasonal_states - 1

            # freq_seasonal
            for _ in range(self._k_freq_seas_states):
                offset += 1
                self.state_cov[series_idx, offset, offset] = _start_params[
                    "freq_seasonal_var"
                ]

        self.obs_cov[0, 0] = 0
        self.initial_value = np.zeros(self.k_states)
        self.initial_covariance = np.eye(self.k_states) * 1e6

        # self.state_cov = [self.state_cov, self.state_cov]
        # self.obs_cov = [self.obs_cov, self.obs_cov]

    def smooth(self):
        kf = EKalmanFilter(
            state_transition=self.transition,
            observation_model=self.design,
            process_noise=self.state_cov,
            observation_noise=self.obs_cov,
        )

        print("state_transition", kf.state_transition.shape)
        print("process_noise", kf.process_noise.shape)
        print("observation_model", kf.observation_model.shape)
        print("observation_noise", kf.observation_noise.shape)
        print("nobs", kf.observation_model.shape[-2])

        return SIMDKalmanFilterResults(
            self.endog,
            kf,
            exog=self.exog,
            initial_value=self.initial_value,
            initial_covariance=self.initial_covariance,
        )

    def __str__(self):
        print(self.transition.shape)
        return (
            "Transition:\n"
            + str(self.transition)
            + "\nDesign:\n"
            + str(self.design)
            + "\nState cov:\n"
            + str(self.state_cov)
            + "\nObs cov:\n"
            + str(self.obs_cov)
        )

    def setup(self):
        """
        Setup the structural time series representation
        """
        # Initialize the ordered sets of parameters
        #         self.parameters = {}
        #         self.parameters_obs_intercept = {}
        #         self.parameters_obs_cov = {}
        #         self.parameters_transition = {}
        #         self.parameters_state_cov = {}

        # Initialize the fixed components of the state space matrices,
        i = 0  # state offset
        j = 0  # state covariance offset

        #         if self.irregular:
        #             self.parameters_obs_cov['irregular_var'] = 1
        if self.level:
            self.design[0, i] = 1.0
            self.transition[i, i] = 1.0
            if self.trend:
                self.transition[i, i + 1] = 1.0
            if self.stochastic_level:
                #                 self.ssm['selection', i, j] = 1.
                #                 self.parameters_state_cov['level_var'] = 1
                j += 1
            i += 1
        if self.trend:
            self.transition[i, i] = 1.0
            if self.stochastic_trend:
                #                 self.ssm['selection', i, j] = 1.
                #                 self.parameters_state_cov['trend_var'] = 1
                j += 1
            i += 1
        if self.seasonal:
            n = self.seasonal_periods - 1
            self.design[0, i] = 1.0
            self.transition[i : i + n, i : i + n] = companion_matrix(
                np.r_[1, [1] * n]
            ).transpose()
            if self.stochastic_seasonal:
                #                 self.ssm['selection', i, j] = 1.
                #                 self.parameters_state_cov['seasonal_var'] = 1
                j += 1
            i += n
        if self.freq_seasonal:
            for ix, h in enumerate(self.freq_seasonal_harmonics):
                # These are the \gamma_jt and \gamma^*_jt terms in D&K (3.8)
                n = 2 * h
                p = self.freq_seasonal_periods[ix]
                lambda_p = 2 * np.pi / float(p)

                t = 0  # frequency transition matrix offset
                for block in range(1, h + 1):
                    # ibid. eqn (3.7)
                    self.design[0, i + t] = 1.0

                    # ibid. eqn (3.8)
                    cos_lambda_block = np.cos(lambda_p * block)
                    sin_lambda_block = np.sin(lambda_p * block)
                    trans = np.array(
                        [
                            [cos_lambda_block, sin_lambda_block],
                            [-sin_lambda_block, cos_lambda_block],
                        ]
                    )
                    trans_s = np.s_[i + t : i + t + 2]
                    self.transition[trans_s, trans_s] = trans
                    t += 2

                    # freq_seasonal is always stochastic

                    j += n
                i += n

        # exog regression
        if self.regression:

            # add exog to the design matrix (3d matrices are a special case in our KF)
            self.design = np.repeat(self.design[np.newaxis, :, :], self.nobs, axis=0)
            self.design[:, 0, i : i + self.k_exog] = self.exog

            self.transition[i : i + self.k_exog, i : i + self.k_exog] = np.eye(
                self.k_exog
            )

            i += self.k_exog
