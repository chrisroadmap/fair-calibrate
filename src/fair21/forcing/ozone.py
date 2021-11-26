"""
Module for ozone forcing
"""

import numpy as np
from ..constants.general import SPECIES_AXIS
from ..constants.gases import BR_ATOMS, CL_ATOMS
from ..defaults.ozone import radiative_efficiency, br_cl_ratio, fractional_release
from ..defaults.forcing import tropospheric_adjustment
from ..defaults.gases import pre_industrial_concentration
from ..defaults.short_lived_forcers import pre_industrial_emissions
from ..defaults import gas_list, slcf_list

# important enough to stand alone
def calculate_eesc(
    concentration,
    pre_industrial_concentration,
    fractional_release,
    cl_atoms,
    br_atoms,
    species_index_mapping,
    br_cl_ratio=br_cl_ratio,
):
    """Calculate equivalent effective stratospheric chlorine.

    Parameters
    ----------
    concentration : ndarray
        concentrations in timestep
    pre_industrial_concentration : ndarray
        pre-industrial concentrations
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
    cfc11_fr = fractional_release[:, species_index_mapping["CFC-11"], :, :]

    eesc_out = (
        cl_atoms * (concentration - pre_industrial_concentration) * fractional_release / cfc11_fr +
        br_cl_ratio * br_atoms * (concentration - pre_industrial_concentration) * fractional_release / cfc11_fr
    ) * cfc11_fr
    return eesc_out


def thornhill_skeie(
    emissions,
    concentration,
    pre_industrial_emissions,
    pre_industrial_concentration,
    fractional_release,
    cl_atoms,
    br_atoms,
    tropospheric_adjustment,
    radiative_efficiency,
    species_index_mapping,
    temperature=0,
    temperature_feedback=-0.037,
    br_cl_ratio=br_cl_ratio,
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
    tropospheric_adjustment : ndarray
        conversion factor from radiative forcing to effective radiative forcing.
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
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """

    array_shape = emissions.shape
    n_scenarios, n_species, n_timesteps, _ = array_shape

    # revisit this if we ever want to dump out intermediate calculations like the feedback strength.
    _erf = np.ones((n_scenarios, 4, n_timesteps, 1)) * np.nan

    # Halogens expressed as EESC
    eesc = calculate_eesc(
        concentration,
        pre_industrial_concentration,
        fractional_release,
        cl_atoms,
        br_atoms,
        species_index_mapping,
        br_cl_ratio=br_cl_ratio,
    )

    _erf[:, 0, ...] = np.nansum(eesc * radiative_efficiency * tropospheric_adjustment, axis=SPECIES_AXIS)

    # Non-Halogens
    # I'm going to say it's OK to hard-code the gases here; we do it for ERF after all.
    o3_species_conc = [
        species_index_mapping["CH4"],
        species_index_mapping["N2O"]
    ]
    o3_species_emis = [
        species_index_mapping["CO"],
        species_index_mapping["VOC"],
        species_index_mapping["NOx"],
    ]

    _erf[:, 1, ...] = np.sum(
        (concentration[:, o3_species_conc, ...] - pre_industrial_concentration[:, o3_species_conc, ...]) *
    radiative_efficiency[:, o3_species_conc, ...], axis=SPECIES_AXIS)

    _erf[:, 2, ...] = np.sum(
        (emissions[:, o3_species_emis, ...] - pre_industrial_emissions[:, o3_species_emis, ...]) *
    radiative_efficiency[:, o3_species_emis, ...], axis=SPECIES_AXIS)

    # Temperature feedback
    _erf[:, 3, ...] = (
        temperature_feedback * temperature * np.sum(_erf[:, :3, ...], axis=SPECIES_AXIS)
    )

    erf_out = np.sum(_erf, axis=SPECIES_AXIS, keepdims=True)

    return erf_out
