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
    #indices_in,
    #index_out,
):
    """
    Calculate effective radiative forcing from a linear relationship of
    emissions, concentrations or forcing.

    Inputs
    ------
    driver : ndarray
        input emissions, concentration or forcing
    baseline_driver : ndarray
        baseline, possibly pre-industrial, emissions, concentration or forcing.
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    radiative_efficiency : ndarray
        radiative efficiency (W m-2 (driver_unit yr-1)-1) of each species.
    indices_in : list of int
        provides a mapping of which species along the SPECIES_AXIS to include
        in the forcing calculation.
    index_out : int
        provides the index that will contain the forcing output.

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2)

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (time, scenario, config, species, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """

    erf_out = np.nansum(
        (
            (emissions - baseline_emissions)
            * radiative_efficiency
        ) * forcing_scaling,
    axis=SPECIES_AXIS, keepdims=True)
    return erf_out
