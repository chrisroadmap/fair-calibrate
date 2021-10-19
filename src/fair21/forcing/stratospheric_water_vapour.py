import numpy as np

from ..defaults.stratospheric_water_vapour import from_methane_fraction

def linear(
    effective_radiative_forcing,
    ratio=from_methane_fraction["AR6"]
):
    """Calculates radiative forcing from oxidation of methane to H2O.

    Stratospheric water vapour forcing follows a linear
    relationship with the CH4 radiative forcing in MAGICC, AR5, and AR6. No
    scientific basis has been found to implement anything different.

    Inputs
    ------
    effective_radiative_forcing : dict or `np.ndarray` or float
        Effective radiative forcing. If dict, the function will look for a key
        called "CH4" and use this. If float or `np.ndarray`, assume input is
        methane ERF.
    ratio : float
        The fraction of methane ERF to be used as stratospheric water vapour
        ERF.

    Returns
    -------
    erf_h2o :
        Effective radiative forcing from stratospheric water vapour.
    """

    erf_ch4 = effective_radiative_forcing
    if isinstance(effective_radiative_forcing, dict):
        erf_ch4 = effective_radiative_forcing['CH4']
    erf_h2o = (ratio * erf_ch4)
    return erf_h2o
