import numpy as np
from statsmodels.tsa.statespace.tools import companion_matrix


class BaseModel:
    """A base for all models which takes care of all initialization
    procedures."""

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

        assert exog is None or exog.ndim in (2, 3)
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
        if exog is None:
            self.k_exog = 0
        elif exog.ndim == 2:
            self.k_exog = exog.shape[1]
        elif exog.ndim == 3:
            self.k_exog = exog.shape[2]
        else:
            raise
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

        # Initialized later
        self.obs_cov = np.array([[0.0]])

        # Initialized later
        self.initial_value = None
        self.initial_covariance = None

        self.setup()

        self.k_endog = 1
        self.selection = np.eye(self.k_states)[:, :, np.newaxis]
        self.obs_intercept = np.array([[0.0]])
        self.state_intercept = np.array([[0.0]] * self.k_states)
        self.time_invariant = self.design.ndim < 3

        """
        A better definition if all matrices are time varying
        self.time_invariant = (
            self.design.shape[2] == 1           and
            self.obs_cov.shape[2] == 1          and
            self.transition.shape[2] == 1       and
            self.selection.shape[2] == 1        and
            self.state_cov.shape[2] == 1)
        """

    def initialize_fixed(self, obs_cov=0, initial_state_cov=1e6):

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

        self.obs_cov[0, 0] = obs_cov
        self.initial_value = np.zeros(self.k_states)
        self.initial_covariance = np.eye(self.k_states) * initial_state_cov

    def initialize_approx_diffuse(self, obs_cov=0, initial_state_cov=1e6):

        #         sigma_epsilon = 2.0 # affects the measurement error
        #         sigma_xi = 1.0 # affects the local level
        #         sigma_omega = 1.0 # affects the seasonality
        #         self.state_cov[0,0] = sigma_xi ** 2
        #         self.state_cov[1,1] = sigma_omega ** 2
        #         self.obs_cov[0,0] = sigma_epsilon ** 2

        from statsmodels.tsa.filters.hp_filter import hpfilter

        for series_idx in range(self.k_series):

            # Eliminate missing data to estimate starting parameters
            endog = self.endog[series_idx, :]
            exog = (
                self.exog[series_idx, ...]
                if self.exog is not None and self.exog.ndim == 3
                else self.exog
            )
            if np.any(np.isnan(endog)):
                mask = ~np.isnan(endog).squeeze()
                endog = endog[mask]
                if exog is not None:
                    # WARN: currently unused
                    exog = exog[mask]

            # Level / trend variances
            # (Use the HP filter to get initial estimates of variances)
            _start_params = {}

            resid, trend1 = hpfilter(endog)

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

        self.obs_cov[0, 0] = obs_cov
        self.initial_value = np.zeros(self.k_states)
        self.initial_covariance = np.eye(self.k_states) * initial_state_cov

        # self.state_cov = [self.state_cov, self.state_cov]
        # self.obs_cov = [self.obs_cov, self.obs_cov]

    def __str__(self):
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
        """Setup the structural time series representation."""
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
            if self.exog.ndim == 2:
                self.design = np.repeat(
                    self.design[np.newaxis, :, :], self.nobs, axis=0
                )
                self.design[:, 0, i : i + self.k_exog] = self.exog
            elif self.exog.ndim == 3:
                self.design = np.repeat(
                    self.design[np.newaxis, :, :], self.nobs, axis=0
                )
                self.design = np.repeat(
                    self.design[np.newaxis, :, :], self.k_series, axis=0
                )
                self.design[:, :, 0, i : i + self.k_exog] = self.exog
            else:
                raise

            self.transition[i : i + self.k_exog, i : i + self.k_exog] = np.eye(
                self.k_exog
            )

            i += self.k_exog

    def filter(self):
        raise NotImplementedError

    def smooth(self):
        raise NotImplementedError
