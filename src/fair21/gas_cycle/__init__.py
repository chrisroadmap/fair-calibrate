"""
Module containing gas cycle functions
"""

import numpy as np

from ..constants import GAS_BOX_AXIS
from ..constants.gases import burden_per_emission, lifetime
from ..defaults.gases import (
    pre_industrial_concentration,
    partition_fraction,
    iirf_horizon,
    iirf_max
)

def calculate_g(
    lifetime,
    partition_fraction,
    iirf_horizon=iirf_horizon,

):
    """Calculate the `g` components of the gas cycle.
    See Leach et al. (2021), eq. (5)
    Parameters
    ----------
    lifetime : ndarray
        atmospheric burden lifetime of the greenhouse gas (yr).
    partition_fraction : ndarray
        proportion of emissions of gas that go into each atmospheric box.
        The sum across the GAS_BOX_AXIS dimension should be 1.
    iirf_horizon : float, default=100
        time horizon (yr) for time integrated impulse response function.
    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    Returns
    -------
    g0 : float
    g1 : float
    """

    g1 = np.sum(partition_fraction * lifetime * (1 - (1 + iirf_horizon/lifetime) * np.exp(-iirf_horizon/lifetime)), axis=GAS_BOX_AXIS, keepdims=True)
    g0 = np.exp(-1 * np.sum(partition_fraction*lifetime*(1 - np.exp(-iirf_horizon/lifetime)), axis=GAS_BOX_AXIS, keepdims=True)/g1)

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
):
    """
    Calculate greenhouse-gas time constant scaling factor.

    Parameters
    ----------
    cumulative_emissions : ndarray
        GtC cumulative emissions since pre-industrial.
    airborne_emissions : ndarray
        GtC total emissions remaining in the atmosphere.
    temperature : ndarray or float
        K temperature anomaly since pre-industrial.
    iirf_0 : ndarray
        pre-industrial time-integrated airborne fraction.
    iirf_cumulative : ndarray
        sensitivity of time-integrated airborne fraction with atmospheric
        carbon stock.
    iirf_temperature : ndarray
        sensitivity of time-integrated airborne fraction with temperature
        anomaly.
    iirf_airborne : ndarray
        sensitivity of time-integrated airborne fraction with airborne
        emissions.
    g0 : ndarray
        parameter for alpha TODO: description
    g1 : ndarray
        parameter for alpha TODO: description
    iirf_max : float
        maximum allowable value to time-integrated airborne fraction

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    alpha : float
        scaling factor for lifetimes
    """

    iirf = iirf_0 + iirf_cumulative * (cumulative_emissions-airborne_emissions) + iirf_temperature * temperature + iirf_airborne * airborne_emissions
    iirf = (iirf>iirf_max) * iirf_max + iirf * (iirf<iirf_max)
    alpha = g0 * np.exp(iirf / g1)
    print(alpha[0, 0, :, :, 0])

    return alpha
