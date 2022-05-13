import numpy as np
import scipy.linalg
import scipy.stats

from .constants import DOUBLING_TIME_1PCT
from .exceptions import IncompatibleConfigError
from .earth_params import earth_radius, seconds_per_year

class EnergyBalanceModel:
    """Energy balance model that converts forcing to temperature.

    The energy balance model is converted to an impulse-response formulation
    (hence the IR part of FaIR) to allow efficient evaluation. The benefits of
    this are increased as once derived, the "layers" of the energy balance
    model do not communicate with each other. The model description can be
    found in references [1]_, [2]_, [3]_ and [4]_.

    References
    ----------

    .. [1] Leach, N. J., Jenkins, S., Nicholls, Z., Smith, C. J., Lynch, J.,
        Cain, M., Walsh, T., Wu, B., Tsutsui, J., and Allen, M. R. (2021).
        FaIRv2.0.0: a generalized impulse response model for climate uncertainty
        and future scenario exploration. Geoscientific Model Development, 14,
        3007–3036

    .. [2] Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
        Estimation of Stochastic Energy Balance Model Parameters, Journal of
        Climate, 33(18), 7909-7926.

    .. [3] Tsutsui (2017): Quantification of temperature response to CO2 forcing
        in atmosphere–ocean general circulation models. Climatic Change, 140,
        287–305

    .. [4] Geoffroy, O., Saint-Martin, D., Bellon, G., Voldoire, A., Olivié,
        D. J. L., & Tytéca, S. (2013). Transient Climate Response in a Two-
        Layer Energy-Balance Model. Part II: Representation of the Efficacy
        of Deep-Ocean Heat Uptake and Validation for CMIP5 AOGCMs, Journal
        of Climate, 26(6), 1859-1876
    """

    def __init__(self, **kwargs):
        """Initialise the EnergyBalanceModel.

        Parameters
        ----------
        **kwargs : dict, optional
            Parameters to run the energy balance model with.

        ocean_heat_capacity : `np.ndarray`
            Ocean heat capacity of each layer (top first), W m-2 yr K-1
        ocean_heat_transfer : `np.ndarray`
            Heat exchange coefficient between ocean layers (top first). The
            first element of this array is akin to the climate feedback
            parameter, with the convention that stabilising feedbacks are
            positive (opposite to most climate sensitivity literature).
            W m-2 K-1
        deep_ocean_efficacy : float
            efficacy of deepest ocean layer. See e.g. [1]_.
        forcing_4co2 : float
            effective radiative forcing from a quadrupling of atmospheric
            CO2 concentrations above pre-industrial.
        stochastic_run : bool
            Activate the stochastic variability component from [2]_.
        sigma_eta : float
            Standard deviation of stochastic forcing component from [2]_.
        sigma_xi : float
            Standard deviation of stochastic disturbance applied to surface
            layer. See [2]_.
        gamma_autocorrelation : float
            Stochastic forcing continuous-time autocorrelation parameter.
            See [2]_.
        seed : int or None
            Random seed to use for stochastic variability.
        timestep : float
            Time interval of the model (yr)

        References
        ----------
        .. [1] Geoffroy, O., Saint-Martin, D., Bellon, G., Voldoire, A., Olivié,
            D. J. L., & Tytéca, S. (2013). Transient Climate Response in a Two-
            Layer Energy-Balance Model. Part II: Representation of the Efficacy
            of Deep-Ocean Heat Uptake and Validation for CMIP5 AOGCMs, Journal
            of Climate, 26(6), 1859-1876

        .. [2] Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
            Estimation of Stochastic Energy Balance Model Parameters, Journal of
            Climate, 33(18), 7909-7926.
        """
        ocean_heat_capacity = kwargs.get('ocean_heat_capacity', np.array([5, 20, 100]))
        self.ocean_heat_transfer = kwargs.get('ocean_heat_transfer', np.array([1.31, 2, 1]))
        self.deep_ocean_efficacy = kwargs.get('deep_ocean_efficacy', 1.2)
        self.forcing_4co2 = kwargs.get('forcing_4co2', 7.86)
        self.stochastic_run = kwargs.get('stochastic_run', False)
        self.sigma_eta = kwargs.get('sigma_eta', 0.5)
        self.sigma_xi = kwargs.get('sigma_xi', 0.5)
        self.gamma_autocorrelation = kwargs.get('gamma_autocorrelation', 2)
        self.seed = kwargs.get('seed', None)
        self.n_temperature_boxes = len(ocean_heat_capacity)
        if len(self.ocean_heat_transfer) != self.n_temperature_boxes:
            raise IncompatibleConfigError("ocean_heat_capacity and ocean_heat_transfer must be arrays of the same shape.")
        self.temperature = kwargs.get('temperature', np.zeros((1, self.n_temperature_boxes + 1)))
        self.n_timesteps = kwargs.get('n_timesteps', 1)
        self.nmatrix = self.n_temperature_boxes + 1
        self.timestep = kwargs.get('timestep', 1)

        # adjust ocean heat capacity to be a rate: units W m-2 K-1
        self.ocean_heat_capacity = ocean_heat_capacity / self.timestep

    def _eb_matrix(self):
        """Define the matrix of differential equations.

        Returns
        -------
        eb_matrix_eigenvalues : `np.ndarray`
            1D array of eigenvalues of the energy balance matrix.
        eb_matrix_eigenvectors : `np.ndarray`
            2D array of eigenvectors (an array of 1D eigenvectors) of the
            energy balance matrix.
        """
        # two box model
        # [x  x]
        # [x  x]

        # three box model
        # [x  x  0]
        # [x  x ex]
        # [0  x  x]

        # four box model
        # [x  x  0  0]
        # [x  x  x  0]
        # [0  x  x ex]
        # [0  0  x  x]

        # put the efficacy of deep ocean in the right place
        # making a vector avoids if statements
        eb_matrix = np.zeros((n_temperature_boxes, n_temperature_boxes))
        epsilon_array = np.ones(n_temperature_boxes)
        epsilon_array[n_temperature_boxes-2] = deep_ocean_efficacy

        # First row
        eb_matrix[0, :2] = [
            -(self.ocean_heat_transfer[0]+epsilon_array[0]*self.ocean_heat_transfer[1])/self.ocean_heat_capacity[0],
            epsilon_array[0]*self.ocean_heat_transfer[1]/self.ocean_heat_capacity[0],
        ]
        # Last row
        eb_matrix[-1, -2:] = [
            self.ocean_heat_transfer[-1]/self.ocean_heat_capacity[-1],
            -self.ocean_heat_transfer[-1]/self.ocean_heat_capacity[-1]
        ]
        # Intermediate rows where n>2
        for row in range(1, n_temperature_boxes-2):
            eb_matrix[row, row-1:row+2] = [
                self.ocean_heat_transfer[row]/self.ocean_heat_capacity[row],
                -(self.ocean_heat_transfer[row]+epsilon_array[row]*self.ocean_heat_transfer[row+1])/self.ocean_heat_capacity[row],
                epsilon_array[row]*self.ocean_heat_transfer[row+1]/self.ocean_heat_capacity[row]
            ]

        # Prepend eb_matrix with stochastic terms if this is a stochastic run: Cummins et al. (2020) eqs. 13 and 14
        eb_matrix = np.insert(eb_matrix, 0, np.zeros(self.n_temperature_boxes), axis=0)
        prepend_col = np.zeros(self.n_temperature_boxes+1)
        prepend_col[0] = -self.gamma_autocorrelation
        prepend_col[1] = 1/self.ocean_heat_capacity[0]
        eb_matrix = np.insert(eb_matrix, 0, prepend_col, axis=1)
        return eb_matrix


    @property
    def eb_matrix_d(self):
        _eb_matrix_d = scipy.linalg.expm(self._eb_matrix())
        return _eb_matrix_d


    def _forcing_vector(self):
        forcing_vector = np.zeros(self.n_temperature_boxes + 1)
        forcing_vector[0] = self.gamma_autocorrelation
        return forcing_vector


    @property
    def forcing_vector_d(self):
        return scipy.linalg.solve(self._eb_matrix(), (self.eb_matrix_d - np.identity(self.n_temperature_boxes + 1)) @ self._forcing_vector())


    @property
    def stochastic_d(self):
        # define stochastic matrix
        _stochastic_d = np.zeros((self.n_timesteps, self.n_temperature_boxes+1))

        # stochastic stuff
        if self.stochastic_run:
            eb_matrix = self._eb_matrix()
            q_mat = np.zeros((self.nmatrix, self.nmatrix))
            q_mat[0,0] = self.sigma_eta**2
            q_mat[1,1] = (self.sigma_xi/self.ocean_heat_capacity[0])**2
            ## use Van Loan (1978) to compute the matrix exponential
            h_mat = np.zeros((self.nmatrix*2, self.nmatrix*2))
            h_mat[:self.nmatrix,:self.nmatrix] = -eb_matrix
            h_mat[:self.nmatrix,self.nmatrix:] = q_mat
            h_mat[self.nmatrix:,self.nmatrix:] = eb_matrix.T
            g_mat = scipy.linalg.expm(h_mat)
            q_mat_d = g_mat[self.nmatrix:,self.nmatrix:].T @ g_mat[:self.nmatrix,self.nmatrix:]
            q_mat_d = q_mat_d.astype(np.float64)
            _stochastic_d = scipy.stats.multivariate_normal.rvs(
                size=self.n_timesteps, mean=np.zeros(self.nmatrix), cov=q_mat_d, random_state=self.seed
            )

        return _stochastic_d

    def impulse_response(self):
        """Converts the energy balance to impulse response."""
        eb_matrix = self._eb_matrix()

        # calculate the eigenvectors and eigenvalues on the energy balance
        # (determininstic) part of the matrix, these are the timescales of responses
        eb_matrix_eigenvalues, eb_matrix_eigenvectors = scipy.linalg.eig(eb_matrix[1:, 1:])
        self.timescales = -1/(np.real(eb_matrix_eigenvalues))
        self.response_coefficients = self.timescales * (eb_matrix_eigenvectors[0,:] * scipy.linalg.inv(eb_matrix_eigenvectors)[:,0]) / self.ocean_heat_capacity[0]


    def emergent_parameters(self, forcing_2co2_4co2_ratio=0.5):
        """Calculates emergent parameters from the energy balance parameters.

        Parameters
        ----------
        forcing_2co2_4co2_ratio : float
            ratio of (effective) radiative forcing converting a quadrupling of
            CO2 to a doubling of CO2.
        """
        # requires impulse response step
        if not hasattr(self, 'timescales'):
            self.impulse_response()
        self.ecs = self.forcing_4co2 * forcing_2co2_4co2_ratio * np.sum(self.response_coefficients)
        self.tcr = self.forcing_4co2 * forcing_2co2_4co2_ratio * np.sum(
            self.response_coefficients*(
                1 - self.timescales/DOUBLING_TIME_1PCT * (
                    1 - np.exp(-DOUBLING_TIME_1PCT/self.timescales)
                )
            )
        )


    def add_forcing(self, forcing, timestep):
        self.forcing = forcing
        self.timestep = timestep
        self.n_timesteps = len(forcing)


    def run(self):
        # internal variables
        forcing_vector = self._forcing_vector()

        # Calculate the matrix exponential
        eb_matrix = self._eb_matrix()
        eb_matrix_d = scipy.linalg.expm(eb_matrix)

        # Solve for temperature
        forcing_vector_d = scipy.linalg.solve(eb_matrix, (eb_matrix_d - np.identity(self.nmatrix)) @ forcing_vector)

        solution = np.zeros((self.n_timesteps, self.nmatrix))
        solution[0, :] = self.temperature[0, :]
        for i in range(1, self.n_timesteps):
            solution[i, :] = eb_matrix_d @ solution[i-1, :] + forcing_vector_d * self.forcing[i-1] + self.stochastic_d[i-1, :]

        self.temperature = solution[:, 1:]
        self.stochastic_forcing = solution[:, 0]
        self.toa_imbalance = self.forcing - self.ocean_heat_transfer[0]*self.temperature[:,0] + (1 - self.deep_ocean_efficacy) * self.ocean_heat_transfer[2] * (self.temperature[:,1] - self.temperature[:,2])
        self.ocean_heat_content_change = np.cumsum(self.toa_imbalance * self.timestep * earth_radius**2 * 4 * np.pi * seconds_per_year)

    def step_temperature(self, temperature_boxes_old, forcing):
        """Timestep the temperature forward.

        Unlike the `run` instance, this increments a single timestep, and should
        prevent inverting a matrix each time.

        Parameters
        ----------
        temperature_boxes_old : ndarray
            array of temperature boxes
        forcing : float
            effective radiative forcing in the timestep

        Returns
        -------
        temperature_boxes_new : ndarray
            array of temperature boxes
        """

        # forcing_vector and eb_matrix should both be determininstic if not running stochastically
        # and we probably can't make this quick if we were
        # actually we probably can...
        temperature_new = self.eb_matrix_d @ temperature_old + self.forcing_vector_d * forcing

        # think we'd be better with step_temperature here
        # OHC now needs to be after the fact
        #self.ocean_heat_content_change = np.cumsum(self.toa_imbalance * np.concatenate(([0], np.diff(self.time)))) * EARTH_RADIUS**2 * 4 * np.pi * SECONDS_PER_YEAR
