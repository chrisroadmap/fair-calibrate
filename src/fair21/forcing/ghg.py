import numpy as np

from ..constants import SPECIES_AXIS
from ..defaults.gases import (
    pre_industrial_concentration,
    radiative_efficiency
)
from ..defaults.forcing import tropospheric_adjustment

def ghg(
    concentration,
    pre_industrial_concentration,
    tropospheric_adjustment,
    radiative_efficiency,
    gas_index_mapping,
    a1 = -2.4785e-07,
    b1 = 0.00075906,
    c1 = -0.0021492,
    d1 = 5.2488,
    a2 = -0.00034197,
    b2 = 0.00025455,
    c2 = -0.00024357,
    d2 = 0.12173,
    a3 = -8.9603e-05,
    b3 = -0.00012462,
    d3 = 0.045194,
    ):
    """Greenhouse gas forcing from CO2, CH4 and N2O including band overlaps.

    Modified Etminan relationship from Meinshausen et al. (2020)
    https://gmd.copernicus.org/articles/13/3571/2020/
    table 3

    Parameters
    ----------
    concentration : ndarray
        concentration of greenhouse gases. "CO2", "CH4" and "N2O" must be
        included in units of [ppm, ppb, ppb]. Other GHGs are units of ppt.
    pre_industrial_concentration : ndarray
        pre-industrial concentration of the gases (see above).
    tropospheric_adjustment : ndarray
        conversion factor from radiative forcing to effective radiative forcing.
    radiative_efficiency : ndarray
        radiative efficiency to use for linear-forcing gases, in W m-2 ppb-1
    gas_index_mapping : dict
        provides a mapping of which gas corresponds to which array index along
        the SPECIES_AXIS.
    a1 : float, default=-2.4785e-07
        fitting parameter (see Meinshausen et al. 2020)
    b1 : float, default=0.00075906
        fitting parameter (see Meinshausen et al. 2020)
    c1 : float, default=-0.0021492
        fitting parameter (see Meinshausen et al. 2020)
    d1 : float, default=5.2488
        fitting parameter (see Meinshausen et al. 2020)
    a2 : float, default=-0.00034197
        fitting parameter (see Meinshausen et al. 2020)
    b2 : float, default=0.00025455
        fitting parameter (see Meinshausen et al. 2020)
    c2 : float, default=-0.00024357
        fitting parameter (see Meinshausen et al. 2020)
    d2 : float, default=0.12173
        fitting parameter (see Meinshausen et al. 2020)
    a3 : float, default=-8.9603e-05
        fitting parameter (see Meinshausen et al. 2020)
    b3 : float, default=-0.00012462
        fitting parameter (see Meinshausen et al. 2020)
    d3 : float, default=0.045194
        fitting parameter (see Meinshausen et al. 2020)

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from greenhouse gases

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    References
    ----------
    [1] Meinshausen et al. 2020
    [2] Myhre et al. 1998
    """
    erf_out = np.ones_like(concentration) * np.nan

    # extracting indices upfront means we're not always searching through array and makes things more readable.
    # expanding the co2_pi array to the same shape as co2 allows efficient conditional indexing
    # TODO: what happens if a scenario does not include all these gases?
    co2 = concentration[:, [gas_index_mapping["CO2"]], ...]
    co2_pi = pre_industrial_concentration[:, [gas_index_mapping["CO2"]], ...] * np.ones_like(co2)
    ch4 = concentration[:, [gas_index_mapping["CH4"]], ...]
    ch4_pi = pre_industrial_concentration[:, [gas_index_mapping["CH4"]], ...]
    n2o = concentration[:, [gas_index_mapping["N2O"]], ...]
    n2o_pi = pre_industrial_concentration[:, [gas_index_mapping["N2O"]], ...]

    # CO2
    ca_max = co2_pi - b1/(2*a1)
    where_central = np.asarray((co2_pi < co2) & (co2 <= ca_max)).nonzero()
    where_low = np.asarray((co2 <= co2_pi)).nonzero()
    where_high = np.asarray((co2 > ca_max)).nonzero()
    alpha_p = np.ones_like(co2) * np.nan
    alpha_p[where_central] = d1 + a1*(co2[where_central] - co2_pi[where_central])**2 + b1*(co2[where_central] - co2_pi[where_central])
    alpha_p[where_low] = d1
    alpha_p[where_high] = d1 - b1**2/(4*a1)
    alpha_n2o = c1*np.sqrt(n2o)
    erf_out[:, [gas_index_mapping["CO2"]], ...] = (alpha_p + alpha_n2o) * np.log(co2/co2_pi) * (1 + tropospheric_adjustment[:, [gas_index_mapping["CO2"]], ...])

    # CH4
    erf_out[:, [gas_index_mapping["CH4"]], ...] = (
        (a3*np.sqrt(ch4) + b3*np.sqrt(n2o) + d3) *
        (np.sqrt(ch4) - np.sqrt(ch4_pi))
    )  * (1 + tropospheric_adjustment[:, [gas_index_mapping["CH4"]], ...])

    # N2O
    erf_out[:, [gas_index_mapping["N2O"]], ...] = (
        (a2*np.sqrt(co2) + b2*np.sqrt(n2o) + c2*np.sqrt(ch4) + d2) *
        (np.sqrt(n2o) - np.sqrt(n2o_pi))
    )  * (1 + tropospheric_adjustment[:, [gas_index_mapping["N2O"]], ...])

    # Then, linear forcing for other gases
    minor_gas_index = list(range(concentration.shape[SPECIES_AXIS]))
    for major_gas in ['CO2', 'CH4', 'N2O']:
        minor_gas_index.remove(gas_index_mapping[major_gas])
    if len(minor_gas_index) > 0:
        erf_out[:, minor_gas_index, ...] = (
            (concentration[:, minor_gas_index, ...] - pre_industrial_concentration[:, minor_gas_index, ...])
            * radiative_efficiency[:, minor_gas_index, ...] * 0.001
        ) * (1 + tropospheric_adjustment[:, minor_gas_index, ...])

    return erf_out


def meinshausen(
    concentration,
    pre_industrial_concentration=pre_industrial_concentration,
    tropospheric_adjustment=tropospheric_adjustment,
    a1 = -2.4785e-07,
    b1 = 0.00075906,
    c1 = -0.0021492,
    d1 = 5.2488,
    a2 = -0.00034197,
    b2 = 0.00025455,
    c2 = -0.00024357,
    d2 = 0.12173,
    a3 = -8.9603e-05,
    b3 = -0.00012462,
    d3 = 0.045194,
    ):
    """Greenhouse gas forcing from CO2, CH4 and N2O including band overlaps.

    Note
    ----
    This is a wrapper version of the array formula provided for convenience.

    Parameters
    ----------
    See `fair.forcing.ghg.ghg`

    Returns
    -------
    effective_radiative_forcing : dict
        effective radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
    """
    scalar_input = False
    gas_index_mapping = {
        "CO2": 0,
        "CH4": 1,
        "N2O": 2,
    }

    # For backward compatibility we allow 3-element arrays
    if isinstance(concentration, np.ndarray):
        temp_dict = {}
        for gas in gas_index_mapping:
            temp_dict[gas] = concentration[gas_index_mapping[gas]]
        concentration = temp_dict
        scalar_input = True

    if isinstance(pre_industrial_concentration, np.ndarray):
        temp_dict = {}
        for gas in gas_index_mapping:
            temp_dict[gas] = pre_industrial_concentration[gas_index_mapping[gas]]
        pre_industrial_concentration = temp_dict

    if np.ndim(concentration['CO2']) == 0:
        n_timestep = 1
        scalar_input = True
    else:
        n_timestep = len(concentration['CO2'])

    concentration_array = np.ones((1, 3, n_timestep, 1)) * np.nan
    pre_industrial_concentration_array = np.ones((1, 3, 1, 1)) * np.nan
    tropospheric_adjustment_array = np.ones((1, 3, 1, 1)) * np.nan
    for gas in gas_index_mapping:
        concentration_array[0, gas_index_mapping[gas], :, 0] = concentration[gas]
        pre_industrial_concentration_array[0, gas_index_mapping[gas], 0, 0] = pre_industrial_concentration[gas]
        tropospheric_adjustment_array[0, gas_index_mapping[gas], 0, 0] = tropospheric_adjustment[gas]

    erf_array = ghg(
        concentration_array,
        pre_industrial_concentration_array,
        tropospheric_adjustment_array,
        np.zeros(3),
        gas_index_mapping,
        a1,
        b1,
        c1,
        d1,
        a2,
        b2,
        c2,
        d2,
        a3,
        b3,
        d3,
        )

    effective_radiative_forcing = {}
    for gas in gas_index_mapping:
         erf_gas = erf_array[0, gas_index_mapping[gas], :, 0]
         if scalar_input:
             effective_radiative_forcing[gas] = erf_gas[0]
         else:
             effective_radiative_forcing[gas] = erf_gas

    return effective_radiative_forcing
