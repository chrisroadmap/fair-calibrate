import numpy as np

from ...constants import SPECIES_AXIS

def calculate_erfari_forcing(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    forcing_scaling,
    radiative_efficiency,
    slcf_indices,
    ghg_indices
):
    """
    Calculate effective radiative forcing from aerosol-radiation interactions.

    Inputs
    ------
    emissions : ndarry
        emissions in timestep
    concentration: ndarray
        concentrations in timestep
    baseline_emissions : ndarray
        pre-industrial emissions
    baseline_concentration : ndarray
        pre-industrial concentrations
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    radiative_efficiency : ndarray
        radiative efficiency (W m-2 (emission_unit yr-1)-1) of each species.
    slcf_indices : list of int
        provides a mapping of which aerosol species corresponds to which emitted
        species index along the SPECIES_AXIS.
    ghg_indices : list of int
        provides a mapping of which aerosol species corresponds to which
        atmospheric GHG concentration along the SPECIES_AXIS.

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

    # zeros because nansum is slow?
    erf_out = np.zeros((emissions.shape[0], emissions.shape[1], emissions.shape[2], emissions.shape[3]))

    # emissions-driven forcers
    erf_out[:, :, :, slcf_indices] = (
        (emissions[:, :, :, slcf_indices, 0] - baseline_emissions[:, :, :, slcf_indices, 0])
        * radiative_efficiency[:, :, :, slcf_indices, 0]
    ) * forcing_scaling[:, :, :, slcf_indices, 0]

    # concentration-driven forcers
    erf_out[:, :, :, ghg_indices] = (
        (emissions[:, :, :, ghg_indices, 0] - baseline_emissions[:, :, :, ghg_indices, 0])
        * radiative_efficiency[:, :, :, ghg_indices, 0]
    ) * forcing_scaling[:, :, :, ghg_indices, 0]

    # in future we can retain contributions from each species. Will need one
    # array index for each species so we don't do this here yet.
    return erf_out.sum(axis=SPECIES_AXIS, keepdims=True)
