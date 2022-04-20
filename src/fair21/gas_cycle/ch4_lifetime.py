"""
Alternative methane lifetime definition that is based on multiple species.
"""

import numpy as np
from ..constants import SPECIES_AXIS
#import warnings

def calculate_alpha_ch4(
    emissions,
    concentration,
    eesc,
    temperature,
    baseline_emissions,
    baseline_concentration,
    normalisation,
    eesc_normalisation,
    ch4_lifetime_chemical_sensitivity,
    ch4_lifetime_eesc_sensitivity,
    ch4_lifetime_temperature_sensitivity,
    slcf_indices,
    ghg_indices,
):

    #print(np.min(1 + temperature * ch4_lifetime_temperature_sensitivity))
    #print(np.argmin(np.squeeze(1 + temperature * ch4_lifetime_temperature_sensitivity)))

    #with warnings.catch_warnings():
    #    warnings.filterwarnings("error")
    #    try:
    log_lifetime_scaling = (
        np.sum(
            np.log(
                1 +
                (emissions[:,:,:,slcf_indices,:]-baseline_emissions[:,:,:,slcf_indices,:])/normalisation[:,:,:,slcf_indices,:]
                * ch4_lifetime_chemical_sensitivity[:,:,:,slcf_indices,:]
            ),
        axis=SPECIES_AXIS, keepdims=True) +
        np.sum(
            np.log(
                1 +
                (concentration[:,:,:,ghg_indices,:]-baseline_concentration[:,:,:,ghg_indices,:])/
                normalisation[:,:,:,ghg_indices,:]
                * ch4_lifetime_chemical_sensitivity[:,:,:,ghg_indices,:],
            ),
        axis=SPECIES_AXIS, keepdims=True) +
        np.log(1 + (eesc/eesc_normalisation * ch4_lifetime_eesc_sensitivity)) +
        np.log(1 + temperature * ch4_lifetime_temperature_sensitivity)
    )
    #    except Warning:
    #        print('stopped')

    return np.exp(log_lifetime_scaling)
