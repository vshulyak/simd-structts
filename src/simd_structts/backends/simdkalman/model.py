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
            exog = np.vstack([self.exog, exog])
        else:
            exog = self.exog
        return self.kf.compute(
            self.endog,
            horizon,
            exog=exog,
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
        )  # .observations.mean

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

        self.mle_regression = False

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

        # correct?
        self.nobs = endog.shape[1]

        # Exogenous component
        #         (self.k_exog, exog) = prepare_exog(exog)
        self.k_exog = exog.shape[1] if exog is not None else 0
        self.exog = exog
        #         print("e1", self.exog)
        #         self.k_exog = 0

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

        self.transition = np.zeros((k_states, k_states))
        self.design = np.zeros((1, k_states))
        self.state_cov = np.zeros((k_states, k_states))
        self.obs_cov = np.array([[0.0]])

        self.initial_value = np.ones(k_states)
        self.initial_covariance = np.eye(4)

        self.setup()

    def initialize_fixed(self):

        #         sigma_epsilon = 2.0 # affects the measurement error
        #         sigma_xi = 1.0 # affects the local level
        #         sigma_omega = 1.0 # affects the seasonality
        #         self.state_cov[0,0] = sigma_xi ** 2
        #         self.state_cov[1,1] = sigma_omega ** 2
        #         self.obs_cov[0,0] = sigma_epsilon ** 2

        offset = 0

        # level
        self.state_cov[offset, offset] = 1

        # trend
        if self.trend:
            offset += 1
            self.state_cov[offset, offset] = 1

        # seasonal
        if self.seasonal:
            offset += 1
            self.state_cov[offset, offset] = 1

            # account for added seasonal components
            offset += self._k_seasonal_states - 1

        # freq_seasonal
        for _ in range(self._k_freq_seas_states):
            offset += 1
            self.state_cov[offset, offset] = 1

        self.obs_cov[0, 0] = 0
        self.initial_value = np.zeros(self.k_states)
        self.initial_covariance = np.eye(self.k_states)

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

                    #                 if self.stochastic_freq_seasonal[ix]:
                    #                     self.selection[i:i + n, j:j + n] = np.eye(n)
                    #                     cov_key = 'freq_seasonal_var_{!r}'.format(ix)
                    #                     self.parameters_state_cov[cov_key] = 1
                    j += n
                i += n

        if self.regression:
            #             if self.mle_regression:
            #                 self.parameters_obs_intercept['reg_coeff'] = self.k_exog
            #             else:
            #             print(self.design.shape)

            #             design = np.repeat(self.design[np.newaxis, :, :], self.nobs,
            #                                axis=0)
            # #             print(design.shape)

            #             self.design = design.transpose()
            # #             print(self.design.shape)

            #             self.design[0, i:i+self.k_exog, :] = (
            #                 self.exog.transpose())

            self.transition[i : i + self.k_exog, i : i + self.k_exog] = np.eye(
                self.k_exog
            )

            i += self.k_exog


#         self.transition = np.ones((2, 2))
#         self.design = np.ones((3,1, 2))
#         self.state_cov = np.random.random((2, 2))
#         self.obs_cov = np.ones((1,1))
