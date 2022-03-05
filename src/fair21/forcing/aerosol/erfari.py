import numpy as np

def calculate_erfari_forcing(
    emissions,
    pre_industrial_emissions,
    forcing_scaling,
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
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
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

    erf_out = np.ones((emissions.shape[0], emissions.shape[1], emissions.shape[2], emissions.shape[3], 0)) * np.nan
    erf_out = (
        (emissions[:, :, :, :, :] - pre_industrial_emissions[:, :, :, :, :])
        * radiative_efficiency[:, :, :, :, :]
    ) * forcing_scaling[:, :, :, :, :]

    return erf_out
