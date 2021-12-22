import numpy as np


def linear(
    emissions,
    pre_industrial_emissions,
    tropospheric_adjustment,
    radiative_efficiency,
    aerosol_index_mapping,
):
    """
    Calculate effective radiative forcing from aerosol-radiation interactions.

    Inputs
    ------
    emissions : ndarray
        input emissions
    pre_industrial_emissions : ndarray
        pre-industrial emissions
    tropospheric_adjustment : ndarray
        conversion factor from radiative forcing to effective radiative forcing.
    radiative_efficiency : ndarray
        radiative efficiency (W m-2 (emission_unit yr-1)-1) of each species.
    aerosol_index_mapping : dict
        provides a mapping of which aerosol species corresponds to which array
        index along the SPECIES_AXIS.

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

    ari_index = list(aerosol_index_mapping.values())
    if len(ari_index) > 0:
        erf_out = np.ones((emissions.shape[0], emissions.shape[1], emissions.shape[2], len(ari_index), 0))
        erf_out = (
            (emissions[:, :, :, ari_index, :] - pre_industrial_emissions[:, :, :, ari_index, :])
            * radiative_efficiency[:, :, :, ari_index, :]
        ) * (1 + tropospheric_adjustment[:, :, :, ari_index, :])

    return erf_out
