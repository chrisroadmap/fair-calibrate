"""
Module for a generic linear emissions to forcing calculation.
"""

import numpy as np
from ..constants import SPECIES_AXIS

def calculate_linear_forcing(
    emissions,
    baseline_emissions,
    forcing_scaling,
    radiative_efficiency,
):
    """
    Calculate effective radiative forcing from aerosol-radiation interactions.
    Inputs
    ------
    emissions : ndarray
        input emissions
    baseline_emissions : ndarray
        baseline, possibly pre-industrial, emissions
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    radiative_efficiency : ndarray
        radiative efficiency (W m-2 (emission_unit yr-1)-1) of each species.
    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from aerosol-radiation interactions
    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (time, scenario, config, species, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """
    erf_out = np.nansum(
        (emissions - baseline_emissions) * radiative_efficiency * forcing_scaling,
        axis=SPECIES_AXIS,
        keepdims=True
    )
    return erf_out
