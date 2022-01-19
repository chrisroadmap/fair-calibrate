"""
Module for ozone forcing
"""

import numpy as np

from ..constants import SPECIES_AXIS

def calculate_eesc(
    concentration,
    baseline_concentration,
    fractional_release,
    cl_atoms,
    br_atoms,
    species_index_mapping,
    br_cl_ratio,
):
    """Calculate equivalent effective stratospheric chlorine.

    Parameters
    ----------
    concentration : ndarray
        concentrations in timestep
    baseline_concentration : ndarray
        baseline, perhaps pre-industrial concentrations
    fractional_release : ndarray
        fractional release describing the proportion of available ODS that
        actually contributes to ozone depletion.
    cl_atoms : ndarray
        Chlorine atoms in each species
    br_atoms : ndarray
        Bromine atoms in each species
    species_index_mapping : dict
        provides a mapping of which gas corresponds to which array index along
        the SPECIES_AXIS.
    br_cl_ratio : float, default=45
        how much more effective bromine is as an ozone depletor than chlorine.

    Returns
    -------
    eesc_out : ndarray
        equivalent effective stratospheric chlorine

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """

    # EESC is in terms of CFC11-eq
    cfc11_fr = fractional_release[:, :, :, [species_index_mapping["CFC-11"]], :]
    eesc_out = (
        cl_atoms * (concentration - baseline_concentration) * fractional_release / cfc11_fr +
        br_cl_ratio * br_atoms * (concentration - baseline_concentration) * fractional_release / cfc11_fr
    ) * cfc11_fr
    return eesc_out


def calculate_ozone_forcing(
    emissions,
    concentration,
    baseline_emissions,
    baseline_concentration,
    fractional_release,
    cl_atoms,
    br_atoms,
    forcing_scaling,
    ozone_radiative_efficiency,
    temperature,
    temperature_feedback,
    br_cl_ratio,
    species_index_mapping,
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

    # Halogen GHGs expressed as EESC
    eesc = calculate_eesc(
        concentration,
        baseline_concentration,
        fractional_release,
        cl_atoms,
        br_atoms,
        species_index_mapping,
        br_cl_ratio,
    )
    _erf[:, :, :, 0, :] = np.nansum(eesc * ozone_radiative_efficiency * forcing_scaling, axis=SPECIES_AXIS)

    # Non-Halogen GHGs, with a concentration-given ozone radiative_efficiency
    o3_species_conc = []
    for species in ["CH4", "N2O"]:
        if species in species_index_mapping:
            o3_species_conc.append(species_index_mapping[species])
    _erf[:, :, :, 1, :] = np.sum(
        (concentration[:, :, :, o3_species_conc, :] - baseline_concentration[:, :, :, o3_species_conc, :]) *
    ozone_radiative_efficiency[:, :, :, o3_species_conc, :], axis=SPECIES_AXIS)

    # Emissions-based SLCF_OZONE_PRECURSORs
    o3_species_emis = []
    for species in ["CO", "VOC", "NOx"]:
        if species in species_index_mapping:
            o3_species_emis.append(species_index_mapping[species])
    _erf[:, :, :, 2, :] = np.sum(
        (emissions[:, :, :, o3_species_emis, :] - baseline_emissions[:, :, :, o3_species_emis, :]) *
    ozone_radiative_efficiency[:, :, :, o3_species_emis, :], axis=SPECIES_AXIS)

    # Temperature feedback
    _erf[:, :, :, [3], :] = (
        temperature_feedback * temperature * np.sum(_erf[:, :, :, :3, :], axis=SPECIES_AXIS, keepdims=True)
    )
    #print(_erf[:, 7, :, 3, :].squeeze())
    #print(temperature.shape) 1 8 66 1
    #print(temperature[:, 7, :, :].squeeze())
    erf_out = np.sum(_erf, axis=SPECIES_AXIS, keepdims=True)
    return erf_out
