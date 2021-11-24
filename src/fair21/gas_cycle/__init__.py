"""
Module containing gas cycle functions
"""

import numpy as np

from ..constants.gases import burden_per_emission, lifetime
from ..defaults.gases import (
    pre_industrial_concentration,
    partition_fraction,
    iirf_horizon,
    iirf_max
)

def calculate_g(
    lifetime,
    partition_fraction=1,
    iirf_horizon=iirf_horizon,

):
    """Calculate the `g` components of the gas cycle.

    See Leach et al. (2021), eq. (5)

    Inputs
    ------
    lifetime : float
        atmospheric burden lifetime of the greenhouse gas (yr).
    partition_fraction : float, default=1 or `np.ndarray` of float
        proportion of emissions of gas that go into each atmospheric box.
        Should be 1 or sum to 1 if array.
    iirf_horizon : float, default=100
        time horizon (yr) for time integrated impulse response function.

    Returns
    -------
    g0 : float
    g1 : float
    """

    g1 = np.sum(partition_fraction * lifetime * (1 - (1 + iirf_horizon/lifetime) * np.exp(-iirf_horizon/lifetime)))
    #g0 = 1/(np.sinh(np.sum(partition_fraction*lifetime*(1 - np.exp(-iirf_horizon/lifetime)), axis=-1)/g1))
    g0 = np.exp(-1 * np.sum(partition_fraction*lifetime*(1 - np.exp(-iirf_horizon/lifetime)), axis=-1)/g1)

    return g0, g1


def calculate_alpha(
    cumulative_emissions,
    airborne_emissions,
    temperature,
    iirf_0,
    iirf_cumulative,
    iirf_temperature,
    iirf_airborne,
    g0,
    g1,
    iirf_max = iirf_max,
    calculate = True
):
    """
    Calculate greenhouse-gas time constant scaling factor.

    Parameters
    ----------
    cumulative_emissions : float
        GtC cumulative emissions since pre-industrial.
    airborne_emissions : float
        GtC total emissions remaining in the atmosphere.
    temperature : float
        K temperature anomaly since pre-industrial.
    iirf_0 : float
        pre-industrial time-integrated airborne fraction.
    iirf_cumulative : float
        sensitivity of time-integrated airborne fraction with atmospheric
        carbon stock.
    iirf_temperature : float
        sensitivity of time-integrated airborne fraction with temperature
        anomaly.
    iirf_airborne : float
        sensitivity of time-integrated airborne fraction with airborne
        emissions.
    g0 : float
        parameter for alpha TODO: description
    g1 : float
        parameter for alpha TODO: description
    iirf_max : float
        maximum allowable value to time-integrated airborne fraction

    Returns
    -------
    alpha : float
        scaling factor for lifetimes
    """

    iirf = iirf_0 + iirf_cumulative * (cumulative_emissions-airborne_emissions) + iirf_temperature * temperature + iirf_airborne * airborne_emissions
    iirf = (iirf>iirf_max) * iirf_max + iirf * (iirf<iirf_max)
    alpha = g0 * np.exp(iirf / g1)

#    # hopefully this additional if does not slow us down
#    if np.isnan(alpha):
#        alpha=1

    return alpha
