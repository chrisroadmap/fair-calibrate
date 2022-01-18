from abc import ABC
from collections.abc import Iterable
import copy
from numbers import Number
import typing
import warnings

import numpy as np

from .constants import (TIME_AXIS, SPECIES_AXIS, GAS_BOX_AXIS)
from .energy_balance_model import EnergyBalanceModel
from .exceptions import (
    DuplicationError,
    IncompatibleConfigError,
    MissingInputError,
    SpeciesMismatchError,
    TimeMismatchError,
    UnexpectedInputError,
    WrongArrayShapeError
)
from .gas_cycle import calculate_alpha
from .gas_cycle.forward import step_concentration
from .structure.top_level import RunConfig, ACIMethod, Category, RunMode, AggregatedCategory
from .structure.scenario_level import Scenario
from .structure.config_level import Config

# each Scenario will contain a list of Species
# each Config will contain settings, as well as options relating to each Species
# FAIR can contain one or more Scenarios and one or more Configs













def check_duplicate_names(things_to_check, name='the list you supplied'):
    output_list = []
    for thing in things_to_check:
        if thing.name in output_list:
            raise DuplicationError(
                f"{thing.name} is duplicated. Please use a unique name for "
                f"each {name}."
            )
        else:
            output_list.append(thing.name)


# from default_configs import CO2, CH4, ...

#@dataclass
#class CO2(GasConfig):
#    species: Species=Species("CO2")
#    ... all the rest
def verify_scenario_consistency(scenarios):
    """Checks to see if all supplied scenarios are self-consistent.

    Parameters
    ----------

    Returns
    -------
    return n_timesteps_first_scenario_species : int
        Number of timesteps that the scenario emissions/concentration/forcing
        is defined with.
    """
    check_type_of_elements(scenarios, Scenario, name="scenarios")
    check_duplicate_names(scenarios, "Scenario")
    n_species_first_scenario = len(scenarios[0].list_of_species)
    n_timesteps_first_scenario_species = None
    for iscen, scenario in enumerate(scenarios):
        if iscen==0:
            species_included_first_scenario = []
            for ispec in range(n_species_first_scenario):
                if n_timesteps_first_scenario_species is None:
                    for attr in ('emissions', 'concentration', 'forcing'):
                        if getattr(scenarios[iscen].list_of_species[ispec], attr) is not None:
                            n_timesteps_first_scenario_species = len(getattr(scenarios[iscen].list_of_species[ispec], attr))
                species_included_first_scenario.append(scenarios[iscen].list_of_species[ispec].species_id.name)
                for attr in ('emissions', 'concentration', 'forcing'):
                    if getattr(scenarios[iscen].list_of_species[ispec], attr) is not None:
                        n_timesteps_this_scenario_species = len(getattr(scenarios[iscen].list_of_species[ispec], attr))
                        if n_timesteps_this_scenario_species != n_timesteps_first_scenario_species:
                            raise TimeMismatchError(
                                f"Each Species in each Scenario must have the same "
                                f"number of timesteps for their emissions, concentration "
                                f"or forcing"
                            )
        species_included = []
        n_species = len(scenarios[iscen].list_of_species)
        for ispec in range(n_species):
            # check for duplicates
            if scenarios[iscen].list_of_species[ispec].species_id.name in species_included:
                raise DuplicationError(
                    f"{scenarios[iscen].list_of_species[ispec].species_id.name} "
                    f"is duplicated in a Scenario."
                )
            species_included.append(scenarios[iscen].list_of_species[ispec].species_id.name)
            for attr in ('emissions', 'concentration', 'forcing'):
                if getattr(scenarios[iscen].list_of_species[ispec], attr) is not None:
                    n_timesteps_this_scenario_species = len(getattr(scenarios[iscen].list_of_species[ispec], attr))
                    if n_timesteps_this_scenario_species != n_timesteps_first_scenario_species:
                        raise TimeMismatchError(
                            f"Each Species in each Scenario must have the same "
                            f"number of timesteps for their emissions, concentration "
                            f"or forcing"
                        )
        if species_included != species_included_first_scenario:
            raise SpeciesMismatchError(
                f"Each Scenario must contain the same list of Species, in the "
                f"same order")
    return n_timesteps_first_scenario_species


def verify_config_consistency(configs):
# TODO: check EBM configs are the same length as each other and agree with RunConfig()
    """Checks to see if all supplied configs are self-consistent."""
    check_type_of_elements(configs, Config, name="configs")
    check_duplicate_names(configs, 'Config')
    # we shouldn't need to do any further checking on the climate_response
    # as this is handled in the @dataclass constructor.
    n_species_first_config = len(configs[0].species_configs)
    for iconf, config in enumerate(configs):
        if iconf==0:
            species_included_first_config = []
            for ispec in range(n_species_first_config):
                species_included_first_config.append(configs[0].species_configs[ispec].species_id.name)
        species_included = []
        n_species = len(configs[iconf].species_configs)
        for ispec in range(n_species):
            species_included.append(configs[iconf].species_configs[ispec].species_id.name)
        if species_included != species_included_first_config:
            raise SpeciesMismatchError(
                f"Each Config must contain the same list of SpeciesConfigs, in the "
                f"same order")


def _check_aci_params(aci_params, aci_method):
    required_params = {
        ACIMethod.SMITH2018: ['scale', 'Sulfur', 'BC+OC'],
        ACIMethod.STEVENS2015: ['scale', 'Sulfur']
    }
    for param in required_params[aci_method]:
        if param not in aci_params:
            raise IncompatibleConfigError(
                f"For aerosol-cloud interactions using the {aci_method}, "
                f"the aci_params in the construction of Config must include "
                f"{required_params[aci_method]}."
            )



def _map_species_scenario_config(scenarios, configs, run_config):
    """Checks to see if species provided in scenarios have associated configs.

    Parameters
    ----------

    Returns
    -------
    species_index_mapping : dict

    Raises
    ------
    SpeciesMismatchError
    DuplicationError
    IncompatibleConfigError
    """
    # at this point we have checked self-consistency, so we only need to check
    # the first element of each.
    species_included_first_config = []
    n_species_first_config = len(configs[0].species_configs)
    for ispec, species in enumerate(configs[0].species_configs):
        species_included_first_config.append(configs[0].species_configs[ispec].species_id)
    species_included_first_scenario = []
    n_species_first_scenario = len(scenarios[0].list_of_species)
    required_aerosol_species = {
        ACIMethod.SMITH2018: [Category.SULFUR, Category.BC, Category.OC],
        ACIMethod.STEVENS2015: [Category.SULFUR],
    }
    aerosol_species_included = []
    aci_desired = False
    for ispec, species in enumerate(scenarios[0].list_of_species):
        species_included_first_scenario.append(scenarios[0].list_of_species[ispec].species_id)
        if scenarios[0].list_of_species[ispec].species_id.category in required_aerosol_species[run_config.aci_method] and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.EMISSIONS:
            aerosol_species_included.append(scenarios[0].list_of_species[ispec].species_id.category)
        if scenarios[0].list_of_species[ispec].species_id.category == Category.AEROSOL_CLOUD_INTERACTIONS:
            aci_desired = True
            aci_index = ispec
    # check config/scenario species consistency
    if species_included_first_config != species_included_first_scenario:
        raise SpeciesMismatchError(
            f"The list of Species provided to Scenario.list_of_species is "
            f"{[species_id.name for species_id in species_included_first_scenario]}. "
            f"This differs from that provided to Config.species_configs "
            f"{[species_id.name for species_id in species_included_first_config]}."
        )
    # check aerosol species provided are consistent with desired indirect forcing mode
    if aerosol_species_included != required_aerosol_species[run_config.aci_method] and aci_desired:
        raise IncompatibleConfigError(
            f"For aerosol-cloud interactions using the {run_config.aci_method}, "
            f"all of {[species_id.name for species_id in required_aerosol_species[run_config.aci_method]]} "
            f"must be provided in the scenario."
        )
    # by the time we get here, we should have checked that scearios and configs species line up
    # and configs where ACI is defined
    # so we just need to check that each config has the correct aci_params
    if aci_desired:
        for config in configs:
            _check_aci_params(config.species_configs[aci_index].aci_params, run_config.aci_method)
    return species_included_first_config


def check_type_of_elements(things_to_check, desired_type, name='the list you supplied'):
    for thing in things_to_check:
        if not isinstance(thing, desired_type):
            raise TypeError(
                f"{name} contains an element of type {type(thing)} "
                f"where it should be a list of {desired_type} objects"
            )

def _make_time_deltas(time):
    time_inner_bounds = 0.5*(time[1:] + time[:-1])
    time_lower_bound = time[0] - (time_inner_bounds[0] - time[0])
    time_upper_bound = time[-1] + (time[-1] - time_inner_bounds[-1])
    time_bounds = np.concatenate(([time_lower_bound], time_inner_bounds, [time_upper_bound]))
    time_deltas = np.diff(time_bounds)
    return time_deltas


def _make_ebm(configs, n_timesteps):
    eb_matrix_d = []
    forcing_vector_d = []
    stochastic_d = []
    for iconf, config in enumerate(configs):
        conf_clim = config.climate_response
        ebm = EnergyBalanceModel(
            ocean_heat_capacity=conf_clim.ocean_heat_capacity,
            ocean_heat_transfer=conf_clim.ocean_heat_transfer,
            deep_ocean_efficacy=conf_clim.deep_ocean_efficacy,
            stochastic_run=conf_clim.stochastic_run,
            sigma_eta=conf_clim.sigma_eta,
            sigma_xi=conf_clim.sigma_xi,
            gamma_autocorrelation=conf_clim.gamma_autocorrelation,
            seed=conf_clim.seed,
            n_timesteps=n_timesteps,
        )
        eb_matrix_d.append(ebm.eb_matrix_d)
        forcing_vector_d.append(ebm.forcing_vector_d)
        stochastic_d.append(ebm.stochastic_d)
    return eb_matrix_d, forcing_vector_d, stochastic_d




def calculate_ghg_forcing(
    concentration,
    pre_industrial_concentration,
    forcing_scaling,
    radiative_efficiency,
    gas_index_mapping,
    a1 = -2.4785e-07,
    b1 = 0.00075906,
    c1 = -0.0021492,
    d1 = 5.2488,
    a2 = -0.00034197,
    b2 = 0.00025455,
    c2 = -0.00024357,
    d2 = 0.12173,
    a3 = -8.9603e-05,
    b3 = -0.00012462,
    d3 = 0.045194,
    ):
    """Greenhouse gas forcing from CO2, CH4 and N2O including band overlaps.

    Modified Etminan relationship from Meinshausen et al. (2020)
    https://gmd.copernicus.org/articles/13/3571/2020/
    table 3

    Parameters
    ----------
    concentration : ndarray
        concentration of greenhouse gases. "CO2", "CH4" and "N2O" must be
        included in units of [ppm, ppb, ppb]. Other GHGs are units of ppt.
    pre_industrial_concentration : ndarray
        pre-industrial concentration of the gases (see above).
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    radiative_efficiency : ndarray
        radiative efficiency to use for linear-forcing gases, in W m-2 ppb-1
    gas_index_mapping : dict
        provides a mapping of which gas corresponds to which array index along
        the SPECIES_AXIS.
    a1 : float, default=-2.4785e-07
        fitting parameter (see Meinshausen et al. 2020)
    b1 : float, default=0.00075906
        fitting parameter (see Meinshausen et al. 2020)
    c1 : float, default=-0.0021492
        fitting parameter (see Meinshausen et al. 2020)
    d1 : float, default=5.2488
        fitting parameter (see Meinshausen et al. 2020)
    a2 : float, default=-0.00034197
        fitting parameter (see Meinshausen et al. 2020)
    b2 : float, default=0.00025455
        fitting parameter (see Meinshausen et al. 2020)
    c2 : float, default=-0.00024357
        fitting parameter (see Meinshausen et al. 2020)
    d2 : float, default=0.12173
        fitting parameter (see Meinshausen et al. 2020)
    a3 : float, default=-8.9603e-05
        fitting parameter (see Meinshausen et al. 2020)
    b3 : float, default=-0.00012462
        fitting parameter (see Meinshausen et al. 2020)
    d3 : float, default=0.045194
        fitting parameter (see Meinshausen et al. 2020)

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from greenhouse gases

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (time, scenario, config, species, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    References
    ----------
    [1] Meinshausen et al. 2020
    [2] Myhre et al. 1998
    """
    erf_out = np.ones_like(concentration) * np.nan

    # extracting indices upfront means we're not always searching through array and makes things more readable.
    # expanding the co2_pi array to the same shape as co2 allows efficient conditional indexing


    co2 = concentration[:, :, :, [gas_index_mapping["CO2"]], ...]
    co2_pi = pre_industrial_concentration[:, :, :, [gas_index_mapping["CO2"]], ...] * np.ones_like(co2)
    ch4 = concentration[:, :, :, [gas_index_mapping["CH4"]], ...]
    ch4_pi = pre_industrial_concentration[:, :, :, [gas_index_mapping["CH4"]], ...]
    n2o = concentration[:, :, :, [gas_index_mapping["N2O"]], ...]
    n2o_pi = pre_industrial_concentration[:, :, :, [gas_index_mapping["N2O"]], ...]

    # CO2
    ca_max = co2_pi - b1/(2*a1)
    where_central = np.asarray((co2_pi < co2) & (co2 <= ca_max)).nonzero()
    where_low = np.asarray((co2 <= co2_pi)).nonzero()
    where_high = np.asarray((co2 > ca_max)).nonzero()
    alpha_p = np.ones_like(co2) * np.nan
    alpha_p[where_central] = d1 + a1*(co2[where_central] - co2_pi[where_central])**2 + b1*(co2[where_central] - co2_pi[where_central])
    alpha_p[where_low] = d1
    alpha_p[where_high] = d1 - b1**2/(4*a1)
    alpha_n2o = c1*np.sqrt(n2o)
    erf_out[:, :, :, [gas_index_mapping["CO2"]], :] = (alpha_p + alpha_n2o) * np.log(co2/co2_pi) * (forcing_scaling[:, :, :, [gas_index_mapping["CO2"]], :])

    # CH4
    erf_out[:, :, :, [gas_index_mapping["CH4"]], :] = (
        (a3*np.sqrt(ch4) + b3*np.sqrt(n2o) + d3) *
        (np.sqrt(ch4) - np.sqrt(ch4_pi))
    )  * (forcing_scaling[:, :, :, [gas_index_mapping["CH4"]], :])

    # N2O
    erf_out[:, :, :, [gas_index_mapping["N2O"]], :] = (
        (a2*np.sqrt(co2) + b2*np.sqrt(n2o) + c2*np.sqrt(ch4) + d2) *
        (np.sqrt(n2o) - np.sqrt(n2o_pi))
    )  * (forcing_scaling[:, :, :, [gas_index_mapping["N2O"]], :])

    # Then, linear forcing for other gases
    minor_gas_index = list(range(concentration.shape[SPECIES_AXIS]))
    for major_gas in ['CO2', 'CH4', 'N2O']:
        minor_gas_index.remove(gas_index_mapping[major_gas])
    if len(minor_gas_index) > 0:
        erf_out[:, :, :, minor_gas_index, :] = (
            (concentration[:, :, :, minor_gas_index, :] - pre_industrial_concentration[:, :, :, minor_gas_index, :])
            * radiative_efficiency[:, :, :, minor_gas_index, :] * 0.001
        ) * (forcing_scaling[:, :, :, minor_gas_index, :])

    return erf_out


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

    ari_index = list(aerosol_index_mapping.values())
    if len(ari_index) > 0:
        erf_out = np.ones((emissions.shape[0], emissions.shape[1], emissions.shape[2], len(ari_index), 0))
        erf_out = (
            (emissions[:, :, :, ari_index, :] - pre_industrial_emissions[:, :, :, ari_index, :])
            * radiative_efficiency[:, :, :, ari_index, :]
        ) * forcing_scaling[:, :, :, ari_index, :]

    return erf_out


def calculate_erfaci_forcing(
    emissions,
    pre_industrial_emissions,
    forcing_scaling,
    scale,
    shape_sulfur,
    shape_bcoc,
    aerosol_index_mapping,
):
    """Calculate effective radiative forcing from aerosol-cloud interactions.

    This uses the relationship to calculate ERFaci described in Smith et al.
    (2021).

    Inputs
    ------
    emissions : ndarray
        input emissions
    pre_industrial_emissions : ndarray
        pre-industrial emissions
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    scale : ndarray
        scaling factor to apply to the logarithm
    shape_sulfur : ndarray
        scale factor for sulfur emissions
    shape_bcoc : ndarray
        scale factor for BC+OC emissions
    radiative_efficiency : ndarray
        radiative efficiency (W m-2 (emission_unit yr-1)-1) of each species.
    aerosol_index_mapping : dict
        provides a mapping of which aerosol species corresponds to which array
        index along the SPECIES_AXIS.

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from aerosol-cloud interactions

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (time, scenario, config, species, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """

    so2 = emissions[:, :, :, [aerosol_index_mapping["Sulfur"]], ...]
    so2_pi = pre_industrial_emissions[:, :, :, [aerosol_index_mapping["Sulfur"]], ...]
    bc = emissions[:, :, :, [aerosol_index_mapping["BC"]], ...]
    bc_pi = pre_industrial_emissions[:, :, :, [aerosol_index_mapping["BC"]], ...]
    oc = emissions[:, :, :, [aerosol_index_mapping["OC"]], ...]
    oc_pi = pre_industrial_emissions[:, :, :, [aerosol_index_mapping["OC"]], ...]
    aci_index = aerosol_index_mapping["Aerosol-Cloud Interactions"]


    # TODO: raise an error if sulfur, BC and OC are not all there
    radiative_effect = -scale * np.log(
        1 + so2/shape_sulfur +
        (bc + oc)/shape_bcoc
    )
    pre_industrial_radiative_effect = -scale * np.log(
        1 + so2_pi/shape_sulfur +
        (bc_pi + oc_pi)/shape_bcoc
    )

    erf_out = (radiative_effect - pre_industrial_radiative_effect) * forcing_scaling
    return erf_out


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


#### MAIN CLASS ####


class FAIR():
    def __init__(
        self,
        scenarios: typing.List[Scenario]=None,
        configs: typing.List[Config]=None,
        time: np.ndarray=None,
        run_config: RunConfig=RunConfig()
    ):
        if isinstance(scenarios, list):
            self.n_timesteps = verify_scenario_consistency(scenarios)
            self.scenarios = scenarios
        elif scenarios is None:
            self.scenarios = []
        else:
            raise TypeError("scenarios should be a list of Scenarios or None")

        if isinstance(configs, list):
            verify_config_consistency(configs)
            self.configs = configs
        elif configs is None:
            self.configs = []
        else:
            raise TypeError("configs should be a list of Configs or None")

        if time is not None:
            self.define_time(time)
        self.run_config = run_config

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)
        verify_scenario_consistency(self.scenarios)

    def remove_scenario(self, scenario: Scenario):
        self.scenarios.remove(scenario)

    def add_config(self, config: Config):
        self.configs.append(config)
        verify_config_consistency(self.configs)

    def remove_config(self, config: Config):
        self.configs.remove(config)

    def define_time(self, time: np.ndarray):
        if not hasattr(time, '__iter__'):
            raise TimeNotIterableError("Time must be an iterable data type")
        self.time = np.asarray(time)

    def _pre_run_checks(self):
        # Check if necessary inputs are defined
        for attr in ('scenarios', 'configs', 'time'):
            if not hasattr(self, attr):
                raise MissingInputError(
                    f"{attr} was not provided when trying to run"
                )
        check_type_of_elements(self.scenarios, Scenario, 'scenarios')
        check_type_of_elements(self.configs, Config, 'configs')
        self.species = _map_species_scenario_config(self.scenarios, self.configs, self.run_config)
        if self.n_timesteps != len(self.time):
            raise TimeMismatchError(
                f"time vector provided is of length {len(self.time)} whereas "
                f"the supplied Scenario inputs are of length "
                f"{self.n_timesteps}."
            )

    def _assign_indices(self):
        # Now that we know that scenarios and configs are consistent, we can
        # allocate array indices to them for running the model. We also define
        # a class level attribute "species".
        self.scenarios_index_mapping = {}
        self.species_index_mapping = {}
        self.configs_index_mapping = {}
        self.ghg_indices = []
        self.ari_indices = []
        self.aci_index = None
        self.ozone_index = None
        #self.config_indices = []
        for ispec, specie in enumerate(self.species):
            self.species_index_mapping[specie.name] = ispec
            if specie.category in AggregatedCategory.GREENHOUSE_GAS:
                self.ghg_indices.append(ispec)
            if specie.category in AggregatedCategory.AEROSOL:
                self.ari_indices.append(ispec)
            if specie.category == Category.AEROSOL_CLOUD_INTERACTIONS:
                self.aci_index = ispec
            if specie.category == Category.OZONE:
                self.ozone_index = ispec
        for iscen, scenario in enumerate(self.scenarios):
            self.scenarios_index_mapping[scenario.name] = iscen
        for iconf, config in enumerate(self.configs):
            self.configs_index_mapping[config.name] = iconf

    def _fill_concentration(self):
        """After the emissions to concentrations step we want to put the concs into each GreenhouseGas"""
        for ispec, species_name in enumerate(self.species_index_mapping):
            if self.species[ispec].category in AggregatedCategory.GREENHOUSE_GAS: # and self.species[ispec].run_mode == RunMode.EMISSIONS: # don't think necessary as should just be replacing with same thing. We want the alpha for CH4 though.
                for iscen, scenario_name in enumerate(self.scenarios_index_mapping):
                    scen_spec = self.scenarios[iscen].list_of_species[ispec]
                    scen_spec.concentration = self.concentration_array[:, iscen, :, ispec, 0]
                    scen_spec.cumulative_emissions = np.squeeze(self.cumulative_emissions_array[:, iscen, :, ispec, 0])
                    with warnings.catch_warnings():
                        # we know about divide by zero possibility
                        warnings.simplefilter('ignore', RuntimeWarning)
                        scen_spec.airborne_fraction = self.airborne_emissions_array[:, iscen, :, ispec, 0] / self.cumulative_emissions_array[:, iscen, :, ispec, 0]
                    scen_spec.airborne_fraction[self.cumulative_emissions_array[:, iscen, :, ispec, 0]==0]=0
                    if self.species[ispec].category == Category.CH4:
                        scen_spec.effective_lifetime = self.alpha_lifetime_array[:, iscen, :, ispec, 0] * self.lifetime_array[:, 0, :, ispec, 0]

    def _fill_forcing(self):
        """Add the forcing as an attribute to each Species and Scenario"""
        for iscen, scenario in enumerate(self.scenarios):
            for ispec, specie in enumerate(self.species):
                scen_spec = self.scenarios[iscen].list_of_species[ispec]
                scen_spec.forcing = self.forcing_array[:, iscen, :, ispec, 0]
            self.scenarios[iscen].forcing = self.forcing_sum_array[:, iscen, :, 0, 0]
            self.scenarios[iscen].stochastic_forcing = self.stochastic_forcing[:, iscen, :]


    def _fill_temperature(self):
        """Add the temperature as an attribute to each Scenario"""
        for iscen, scenario in enumerate(self.scenarios):
            self.scenarios[iscen].temperature_layers = self.temperature[:, iscen, :, 0, :]
            self.scenarios[iscen].temperature = self.temperature[:, iscen, :, 0, 0]

    def _initialise_arrays(self, n_timesteps, n_scenarios, n_configs, n_species, aci_method):
        self.emissions_array = np.ones((n_timesteps, n_scenarios, n_configs, n_species, 1)) * np.nan
        self.concentration_array = np.ones((n_timesteps, n_scenarios, n_configs, n_species, 1)) * np.nan
        self.forcing_sum_array = np.ones((n_timesteps, n_scenarios, n_configs, 1, 1)) * np.nan
        self.forcing_array = np.ones((n_timesteps, n_scenarios, n_configs, n_species, 1)) * np.nan
        self.g0_array = np.ones((1, n_scenarios, n_configs, n_species, 1)) * np.nan
        self.g1_array = np.ones((1, n_scenarios, n_configs, n_species, 1)) * np.nan
        self.alpha_lifetime = np.ones((1, n_scenarios, n_configs, n_species, 1))
        self.airborne_emissions = np.zeros((1, n_scenarios, n_configs, n_species, 1))
        self.gas_boxes = np.zeros((1, n_scenarios, n_configs, n_species, self.run_config.n_gas_boxes))
        self.iirf_0_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.iirf_cumulative_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.iirf_temperature_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.iirf_airborne_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.burden_per_emission_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.lifetime_array = np.zeros((1, 1, n_configs, n_species, self.run_config.n_gas_boxes))
        self.baseline_emissions_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.baseline_concentration_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.partition_fraction_array = np.zeros((1, 1, n_configs, n_species, self.run_config.n_gas_boxes))
        self.natural_emissions_adjustment_array = np.zeros((1, 1, n_configs, n_species, 1))
        self.radiative_efficiency_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.forcing_scaling_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.efficacy_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.fractional_release_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.br_atoms_array = np.ones((1, 1, 1, n_species, 1)) * np.nan
        self.cl_atoms_array = np.ones((1, 1, 1, n_species, 1)) * np.nan
        self.erfari_emissions_to_forcing_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.erfaci_scale_array = np.ones((1, 1, n_configs, 1, 1)) * np.nan
        self.erfaci_shape_sulfur_array = np.ones((1, 1, n_configs, 1, 1)) * np.nan
        self.erfaci_shape_bcoc_array = np.ones((1, 1, n_configs, 1, 1)) * np.inf
        self.ozone_radiative_efficiency_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        # TODO: make a more general temperature-forcing feedback for all species
        self.forcing_temperature_feedback_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        # TODO: start from non-zero temperature
        self.temperature = np.ones((n_timesteps, n_scenarios, n_configs, 1, self.run_config.n_temperature_boxes)) * np.nan

        for ispec, species_name in enumerate(self.species_index_mapping):
            for iconf, config_name in enumerate(self.configs_index_mapping):
                conf_spec = self.configs[iconf].species_configs[ispec]
                self.forcing_scaling_array[:, 0, iconf, ispec, 0] = (1+conf_spec.tropospheric_adjustment) * conf_spec.scale
                self.efficacy_array[:, 0, iconf, ispec, 0] = conf_spec.efficacy
                self.baseline_emissions_array[:, :, iconf, ispec, :] = conf_spec.baseline_emissions
                self.forcing_temperature_feedback_array[:, :, iconf, ispec, :] = conf_spec.forcing_temperature_feedback
                if self.species[ispec].category in AggregatedCategory.GREENHOUSE_GAS:
                    partition_fraction = np.asarray(conf_spec.partition_fraction)
                    if np.ndim(partition_fraction) == 1:
                        self.partition_fraction_array[:, :, iconf, ispec, :] = conf_spec.partition_fraction
                    else:
                        self.partition_fraction_array[:, :, iconf, ispec, 0] = conf_spec.partition_fraction
                    self.lifetime_array[:, :, iconf, ispec, :] = conf_spec.lifetime
                    self.iirf_0_array[:, :, iconf, ispec, :] = conf_spec.iirf_0
                    self.iirf_cumulative_array[:, :, iconf, ispec, :] = conf_spec.iirf_cumulative
                    self.iirf_temperature_array[:, :, iconf, ispec, :] = conf_spec.iirf_temperature
                    self.iirf_airborne_array[:, :, iconf, ispec, :] = conf_spec.iirf_airborne
                    self.burden_per_emission_array[:, :, iconf, ispec, :] = conf_spec.burden_per_emission
                    self.radiative_efficiency_array[:, :, iconf, ispec, :] = conf_spec.radiative_efficiency
                    self.g0_array[:, :, iconf, ispec, :] = conf_spec.g0
                    self.g1_array[:, :, iconf, ispec, :] = conf_spec.g1
                    self.baseline_concentration_array[:, :, iconf, ispec, :] = conf_spec.baseline_concentration
                    self.natural_emissions_adjustment_array[:, :, iconf, ispec, 0] = conf_spec.natural_emissions_adjustment
                if self.species[ispec].category in AggregatedCategory.HALOGEN:  # TODO: probably needs similar to above here.
                    self.fractional_release_array[:, :, iconf, ispec, 0] = conf_spec.fractional_release
                    self.br_atoms_array[:, :, :, ispec, 0] = conf_spec.br_atoms
                    self.cl_atoms_array[:, :, :, ispec, 0] = conf_spec.cl_atoms
                if self.species[ispec].category in AggregatedCategory.AEROSOL:
                    self.erfari_emissions_to_forcing_array[:, 0, iconf, ispec, :] = conf_spec.erfari_emissions_to_forcing
                if self.species[ispec].category == Category.AEROSOL_CLOUD_INTERACTIONS:
                    self.erfaci_scale_array[0, 0, iconf, 0, 0] = conf_spec.aci_params['scale']
                    self.erfaci_shape_sulfur_array[0, 0, iconf, 0, 0] = conf_spec.aci_params['Sulfur']
                    if aci_method==ACIMethod.SMITH2018:
                        self.erfaci_shape_bcoc_array[0, 0, iconf, 0, 0] = conf_spec.aci_params['BC+OC']
                if self.species[ispec].category in AggregatedCategory.OZONE_PRECURSOR:
                    self.ozone_radiative_efficiency_array[0, 0, iconf, ispec, 0] = conf_spec.ozone_radiative_efficiency
            for iscen, scenario_name in enumerate(self.scenarios_index_mapping):
                scen_spec = self.scenarios[iscen].list_of_species[ispec]
                if self.species[ispec].run_mode == RunMode.EMISSIONS:
                    self.emissions_array[:, iscen, :, ispec, 0] = scen_spec.emissions[:, None]
                if self.species[ispec].run_mode == RunMode.CONCENTRATION:
                    self.concentration_array[:, iscen, :, ispec, 0] = scen_spec.concentration[:, None]
                if self.species[ispec].run_mode == RunMode.FORCING:
                    self.forcing_array[:, iscen, :, ispec, 0] = scen_spec.forcing[:, None]

        self.cumulative_emissions_array = np.cumsum(self.emissions_array * self.time_deltas[:, None, None, None, None], axis=TIME_AXIS)
        self.alpha_lifetime_array = np.ones((n_timesteps, n_scenarios, n_configs, n_species, 1))
        self.airborne_emissions_array = np.zeros((n_timesteps, n_scenarios, n_configs, n_species, 1))
        self.stochastic_forcing = np.ones((n_timesteps, n_scenarios, n_configs)) * np.nan

#### MAIN RUN ####

    def run(self):
        self._pre_run_checks()
        self._assign_indices()
        self.time_deltas = _make_time_deltas(self.time)

        n_species = len(self.species_index_mapping)
        n_configs = len(self.configs_index_mapping)
        n_scenarios = len(self.scenarios_index_mapping)
        n_timesteps = self.n_timesteps

        # from this point onwards, we lose a bit of the clean OO-style and go
        # back to prodecural programming, which is a ton quicker.
        self._initialise_arrays(n_timesteps, n_scenarios, n_configs, n_species, self.run_config.aci_method)
        # move to initialise_arrays?
        gas_boxes = np.zeros((1, n_scenarios, n_configs, n_species, self.run_config.n_gas_boxes))
        temperature_boxes = np.zeros((1, n_scenarios, n_configs, 1, self.run_config.n_temperature_boxes+1))

        # initialise the energy balance model and get critical vectors
        # which itself needs to be run once per "config" and dimensioned correctly
        eb_matrix_d, forcing_vector_d, stochastic_d = _make_ebm(self.configs, n_timesteps)

        # Main loop
        for i_timestep in range(n_timesteps):
            # 1. ghg emissions to concentrations
            alpha_lifetime_array = calculate_alpha(
                self.cumulative_emissions_array[[i_timestep], ...],
                self.airborne_emissions_array[[i_timestep-1], ...],
                temperature_boxes[:, :, :, :, 1:2],
                self.iirf_0_array,
                self.iirf_cumulative_array,
                self.iirf_temperature_array,
                self.iirf_airborne_array,
                self.g0_array,
                self.g1_array,
                self.run_config.iirf_max
            )
#            alpha_lifetime_array[np.isnan(alpha_lifetime_array)]=1  # CF4 seems to have an issue. Should we raise warning?
            self.concentration_array[[i_timestep], ...], gas_boxes, self.airborne_emissions_array[[i_timestep], ...] = step_concentration(
                self.emissions_array[[i_timestep], ...],
                gas_boxes,
                self.airborne_emissions_array[[i_timestep-1], ...],
                self.burden_per_emission_array,
                self.lifetime_array,
                alpha_lifetime=alpha_lifetime_array,
                pre_industrial_concentration=self.baseline_concentration_array,
                timestep=self.time_deltas[i_timestep],
                partition_fraction=self.partition_fraction_array,
                natural_emissions_adjustment=self.natural_emissions_adjustment_array,
            )
            self.alpha_lifetime_array[[i_timestep], ...] = alpha_lifetime_array

            # 2. concentrations to emissions for ghg emissions:
            # TODO:

            # 3. Greenhouse gas concentrations to forcing
            self.forcing_array[i_timestep:i_timestep+1, :, :, self.ghg_indices] = calculate_ghg_forcing(
                self.concentration_array[[i_timestep], ...],
                self.baseline_concentration_array,
                self.forcing_scaling_array,
                self.radiative_efficiency_array,
                self.species_index_mapping
            )[0:1, :, :, self.ghg_indices, :]

            # 4. aerosol emissions to forcing
            self.forcing_array[i_timestep:i_timestep+1, :, :, self.ari_indices, :] = calculate_erfari_forcing(
                self.emissions_array[[i_timestep], ...],
                self.baseline_emissions_array,
                self.forcing_scaling_array,
                self.erfari_emissions_to_forcing_array,
                self.species_index_mapping
            )[0:1, :, :, self.ari_indices, :]

            if self.aci_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, self.aci_index, :] = calculate_erfaci_forcing(
                    self.emissions_array[[i_timestep], ...],
                    self.baseline_emissions_array,
                    self.forcing_scaling_array,
                    self.erfaci_scale_array,
                    self.erfaci_shape_sulfur_array,
                    self.erfaci_shape_bcoc_array,
                    self.species_index_mapping
                )[0:1, :, :, self.aci_index, :]

            # 5. ozone emissions and concentrations to forcing
            if self.ozone_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, [self.ozone_index], :] = calculate_ozone_forcing(
                    self.emissions_array[[i_timestep], ...],
                    self.concentration_array[[i_timestep], ...],
                    self.baseline_emissions_array,
                    self.baseline_concentration_array,
                    self.fractional_release_array,
                    self.cl_atoms_array,
                    self.br_atoms_array,
                    self.forcing_scaling_array,
                    self.ozone_radiative_efficiency_array,
                    temperature_boxes[:, :, :, :, 1:2],
                    self.forcing_temperature_feedback_array[:, :, :, [self.ozone_index], :],
                    self.run_config.br_cl_ratio,
                    self.species_index_mapping
                )


            # contrails here
            # BC on snow here
            # strat water vapour here
            # land use here
            # solar here
            # volcanic here

            # 99. sum up all of the forcing calculated previously
            self.forcing_sum_array[[i_timestep], ...] = np.nansum(
                self.forcing_array[[i_timestep], ...], axis=SPECIES_AXIS, keepdims=True
            )
            efficacy_adjusted_forcing=np.nansum(
                self.forcing_array[[i_timestep], ...]*self.efficacy_array, axis=SPECIES_AXIS, keepdims=True
            )

            # 100. run the energy balance model
            # TODO: skip if temperature prescribed
            # TODO: remove loop
            for iscen, scenario in enumerate(self.scenarios):
                for iconf, config in enumerate(self.configs):
                    temperature_boxes[0, iscen, iconf, 0, :] = (
                        eb_matrix_d[iconf] @ temperature_boxes[0, iscen, iconf, 0, :] +
                        forcing_vector_d[iconf] * efficacy_adjusted_forcing[0, iscen, iconf, 0, 0] +
                        stochastic_d[iconf][i_timestep, :]
                    )
                    self.temperature[i_timestep, iscen, iconf, :, :] = temperature_boxes[0, iscen, iconf, 0, 1:]
                    self.stochastic_forcing[i_timestep, iscen, iconf] = temperature_boxes[0, iscen, iconf, 0, 0]

        self._fill_concentration()
        self._fill_forcing()
        self._fill_temperature()








### main
#for i in range(751):
