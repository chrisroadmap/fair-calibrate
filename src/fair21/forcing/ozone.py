"""
Module for ozone forcing
"""

import numpy as np

from ..constants import SPECIES_AXIS


def thornhill2021(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    eesc,
    forcing_scaling,
    ozone_radiative_efficiency,
    temperature,
    temperature_feedback,
    slcf_indices,
    ghg_indices
):

    """Determines ozone effective radiative forcing.

    Calculates total ozone forcing from precursor emissions and
    concentrations based on AerChemMIP and CMIP6 Historical behaviour in
    Skeie et al. (2020) and Thornhill et al. (2021).

    In this hard-coded treatment, ozone forcing depends on concentrations of
    CH4, N2O, ozone-depleting halogens, and emissions of CO, NVMOC and NOx,
    but any combination of emissions and concentrations are allowed.

    Parameters
    ----------
    emissions : ndarry
        emissions in timestep
    concentration: ndarray
        concentrations in timestep
    pre_industrial_emissions : ndarray
        pre-industrial emissions
    pre_industrial_concentration : ndarray
        pre-industrial concentrations
    fractional_release : ndarray
        fractional release describing the proportion of available ODS that
        actually contributes to ozone depletion.
    cl_atoms : ndarray
        Chlorine atoms in each species
    br_atoms : ndarray
        Bromine atoms in each species
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    radiative_efficiency : ndarray
        the radiative efficiency at which ozone precursor emissions or
        concentrations are converted to ozone radiative forcing. The unit is
        W m-2 (<native emissions or concentration unit>)-1, where the
        emissions unit is usually Mt/yr for short-lived forcers, ppb for CH4
        and N2O concentrations, and ppt for halogenated species. Note this is
        not the same radiative efficiency that is used in the ghg forcing.
    species_index_mapping : ndarray
        provides a mapping of which gas corresponds to which array index along
        the SPECIES_AXIS.
    temperature : ndarray or float
        global mean surface temperature anomaly used to calculate the feedback.
        In the forward model this will be one timestep behind; a future TODO
        could be to iterate this.
    temperature_feedback : float
        temperature feedback on ozone forcing (W m-2 K-1)
    br_cl_ratio : float, default=45
        how much more effective bromine is as an ozone depletor than chlorine.
    cfc_11_index : int or None
        array index along SPECIES_AXIS corresponding to CFC-11.
    halogen_indices : list of int
        provides a mapping of which halogen species corresponds to which
        index along the SPECIES_AXIS.
    slcf_indices : list of int
        provides a mapping of which aerosol species corresponds to which emitted
        species index along the SPECIES_AXIS.
    ghg_indices : list of int
        provides a mapping of which aerosol species corresponds to which
        atmospheric GHG concentration along the SPECIES_AXIS.

    Returns
    -------
    erf_ozone : dict
        ozone forcing due to each component, and in total.

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (time, scenario, config, species, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """

    array_shape = emissions.shape
    n_timesteps, n_scenarios, n_configs, n_species, _ = array_shape

    # revisit this if we ever want to dump out intermediate calculations like the feedback strength.
    _erf = np.ones((n_timesteps, n_scenarios, n_configs, 4, 1)) * np.nan

    # Halogen GHGs expressed as EESC precalculated
    _erf[:, :, :, 0, :] = np.nansum(eesc * ozone_radiative_efficiency * forcing_scaling, axis=SPECIES_AXIS)

    # Non-Halogen GHGs, with a concentration-given ozone radiative_efficiency
    _erf[:, :, :, 1, :] = np.sum(
        (concentration[:, :, :, ghg_indices, :] - baseline_concentration[:, :, :, ghg_indices, :]) *
    ozone_radiative_efficiency[:, :, :, ghg_indices, :], axis=SPECIES_AXIS)

    # Emissions-based precursors
    _erf[:, :, :, 2, :] = np.sum(
        (emissions[:, :, :, slcf_indices, :] - baseline_emissions[:, :, :, slcf_indices, :]) *
    ozone_radiative_efficiency[:, :, :, slcf_indices, :], axis=SPECIES_AXIS)

    # Temperature feedback
    _erf[:, :, :, [3], :] = (
        temperature_feedback * temperature * np.sum(_erf[:, :, :, :3, :], axis=SPECIES_AXIS, keepdims=True)
    )
    erf_out = np.sum(_erf, axis=SPECIES_AXIS, keepdims=True)
    return erf_out
