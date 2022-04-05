"""
Alternative methane lifetime definition that is based on multiple species.
"""

import numpy as np
from ..constants import SPECIES_AXIS

def calculate_alpha_ch4(
    emissions,
    concentration,
    temperature,
    baseline_emissions,
    baseline_concentration,
    normalisation,
    ch4_lifetime_chemical_sensitivity,
    ch4_lifetime_temperature_sensitivity,
    slcf_indices,
    ghg_indices,
):

    #for species in ['CH4', 'N2O', 'VOC', 'HFC', 'NOx', 'temp']:
    print(normalisation[0,0,0,:,0])
    print(ch4_lifetime_chemical_sensitivity[0,0,0,:,0])
    print(ch4_lifetime_temperature_sensitivity[0,0,0,:,0])
    log_lifetime_scaling = (
        np.log(
            1 +
            np.sum(
                (emissions[:,:,:,slcf_indices,:]-baseline_emissions[:,:,:,slcf_indices,:])/normalisation[:,:,:,slcf_indices,:]
                * ch4_lifetime_chemical_sensitivity[:,:,:,slcf_indices,:], axis=SPECIES_AXIS, keepdims=True
            ) +
            np.sum(
                (concentration[:,:,:,ghg_indices,:]-baseline_concentration[:,:,:,ghg_indices,:])/
                normalisation[:,:,:,ghg_indices,:]
                * ch4_lifetime_chemical_sensitivity[:,:,:,ghg_indices,:], axis=SPECIES_AXIS, keepdims=True
            ) +
            temperature * ch4_lifetime_temperature_sensitivity
        )
    )

    return np.exp(log_lifetime_scaling)
