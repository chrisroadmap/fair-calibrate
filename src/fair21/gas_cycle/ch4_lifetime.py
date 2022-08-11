"""
Alternative methane lifetime definition that is based on multiple species.
"""

import numpy as np
from ..constants import SPECIES_AXIS
#import warnings

def calculate_alpha_ch4(
    emissions,
    concentration,
    temperature,
    baseline_emissions,
    baseline_concentration,
    ch4_lifetime_chemical_sensitivity,
    ch4_lifetime_temperature_sensitivity,
    emissions_indices,
    concentration_indices,
):

    log_lifetime_scaling = (
        np.sum(
            np.log(
                1 +
                (emissions[..., emissions_indices]-baseline_emissions[..., emissions_indices])
                * ch4_lifetime_chemical_sensitivity[..., emissions_indices]
            ),
        axis=SPECIES_AXIS, keepdims=True) +
        np.sum(
            np.log(
                1 +
                (concentration[..., concentration_indices]-baseline_concentration[..., concentration_indices])
                * ch4_lifetime_chemical_sensitivity[..., concentration_indices],
            ),
        axis=SPECIES_AXIS, keepdims=True) +
        np.log(1 + temperature * ch4_lifetime_temperature_sensitivity)
    )

    return np.exp(log_lifetime_scaling)
