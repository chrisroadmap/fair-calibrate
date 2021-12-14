"""
Module containing gas cycle functions
"""

import numpy as np

from ..constants import GAS_BOX_AXIS
from ..defaults.gases import iirf_max


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
    temperature : float
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

    return alpha
