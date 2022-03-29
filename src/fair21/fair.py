import copy
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
from .forcing.aerosol.erfari import calculate_erfari_forcing
from .forcing.aerosol.erfaci import calculate_erfaci_forcing, _check_aci_params
from .forcing.ghg import calculate_ghg_forcing
from .forcing.linear import calculate_linear_forcing
from .forcing.ozone import calculate_ozone_forcing
from .gas_cycle import calculate_alpha
from .gas_cycle.forward import step_concentration
from .gas_cycle.inverse import unstep_concentration
from .structure.top_level import RunConfig, ACIMethod, Category, RunMode, AggregatedCategory
from .structure.scenario_level import Scenario
from .structure.config_level import Config

def _check_duplicate_names(things_to_check, name='the list you supplied'):
    output_list = []
    for thing in things_to_check:
        if thing.name in output_list:
            raise DuplicationError(
                f"{thing.name} is duplicated. Please use a unique name for "
                f"each {name}."
            )
        else:
            output_list.append(thing.name)

def _check_type_of_elements(things_to_check, desired_type, name='the list you supplied'):
    for thing in things_to_check:
        if not isinstance(thing, desired_type):
            raise TypeError(
                f"{name} contains an element of type {type(thing)} "
                f"where it should be a list of {desired_type} objects"
            )

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

def _make_time_deltas(time):
    time_inner_bounds = 0.5*(time[1:] + time[:-1])
    time_lower_bound = time[0] - (time_inner_bounds[0] - time[0])
    time_upper_bound = time[-1] + (time[-1] - time_inner_bounds[-1])
    time_bounds = np.concatenate(([time_lower_bound], time_inner_bounds, [time_upper_bound]))
    time_deltas = np.diff(time_bounds)
    return time_deltas

def _map_species_scenario_config(scenarios, configs, run_config):
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
    contrails_from_emissions_desired = False
    nox_aviation_emissions_supplied = False
    h2o_stratospheric_desired = False
    ch4_supplied = False
    land_use_desired = False
    co2_desired = False
    co2_afolu_supplied = False
    co2_ffi_supplied = False
    # For species that depend on other species in certain RunModes, check everything that we need is present.
    for ispec, species in enumerate(scenarios[0].list_of_species):
        species_included_first_scenario.append(scenarios[0].list_of_species[ispec].species_id)
        if scenarios[0].list_of_species[ispec].species_id.category in required_aerosol_species[run_config.aci_method] and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.EMISSIONS:
            aerosol_species_included.append(scenarios[0].list_of_species[ispec].species_id.category)
        if scenarios[0].list_of_species[ispec].species_id.category == Category.AEROSOL_CLOUD_INTERACTIONS:
            aci_desired = True
            aci_index = ispec
        if scenarios[0].list_of_species[ispec].species_id.category == Category.CONTRAILS and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.FROM_OTHER_SPECIES:
            contrails_from_emissions_desired = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.NOX_AVIATION and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.EMISSIONS:
            nox_aviation_emissions_supplied = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.H2O_STRATOSPHERIC and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.FROM_OTHER_SPECIES:
            h2o_stratospheric_desired = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.CH4:
            ch4_supplied = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.LAND_USE and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.FROM_OTHER_SPECIES:
            land_use_desired = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.CO2 and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.FROM_OTHER_SPECIES:
            co2_desired = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.CO2_AFOLU and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.EMISSIONS:
            co2_afolu_supplied = True
        if scenarios[0].list_of_species[ispec].species_id.category == Category.CO2_FFI and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.EMISSIONS:
            co2_ffi_supplied = True
    # check config/scenario species consistency
    # TODO: this error message is quite helpful, but could be further improved by
    # pointing out exactly where they differ.
    if species_included_first_config != species_included_first_scenario:
        # TODO: make better formatted error message
        raise SpeciesMismatchError(species_included_first_scenario, species_included_first_config)
#            f"The list of Species provided to Scenario.list_of_species is: (name, category, run_mode)\n"
#            f"{[(species_id.name, species_id.category.name, species_id.run_mode.name) for species_id in species_included_first_scenario]}. "
#            f"\n\n"
#            f"This differs from that provided to Config.species_configs: (name, category, run_mode)\n"
#            f"{[(species_id.name, species_id.category.name, species_id.run_mode.name) for species_id in species_included_first_config]}."
#        )
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
    # if contrail forcing from emissions is desired, we need aviation NOx emissions
    if contrails_from_emissions_desired and not nox_aviation_emissions_supplied:
        raise IncompatibleConfigError(
            f"For contrails forcing from emissions, aviation NOx emissions "
            f"must be supplied."
        )
    # if stratospheric water vapour desired, we need methane to exist in the scenario
    if h2o_stratospheric_desired and not ch4_supplied:
        raise IncompatibleConfigError(
            f"For stratospheric water vapour forcing, CH4 emissions, "
            f"concentrations, or forcing must be supplied."
        )
    # CO2 emissions from AFOLU and FFI need to be provided
    if co2_desired and not co2_ffi_supplied and not co2_afolu_supplied:
        raise IncompatibleConfigError(
            f"For CO2 emissions-driven, CO2 emissions from FFI and AFOLU must "
            f"both be supplied."
        )
    return species_included_first_config


def _verify_config_consistency(configs):
# TODO: check EBM configs are the same length as each other and agree with RunConfig()
    _check_type_of_elements(configs, Config, name="configs")
    _check_duplicate_names(configs, 'Config')
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


def _verify_scenario_consistency(scenarios):
    _check_type_of_elements(scenarios, Scenario, name="scenarios")
    _check_duplicate_names(scenarios, "Scenario")
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


class FAIR():
    def __init__(
        self,
        scenarios: typing.List[Scenario]=None,
        configs: typing.List[Config]=None,
        time: np.ndarray=None,
        run_config: RunConfig=RunConfig()
    ):
        if isinstance(scenarios, list):
            self.n_timesteps = _verify_scenario_consistency(scenarios)
            self.scenarios = scenarios
        elif scenarios is None:
            self.scenarios = []
        else:
            raise TypeError("scenarios should be a list of Scenarios or None")

        if isinstance(configs, list):
            _verify_config_consistency(configs)
            self.configs = configs
        elif configs is None:
            self.configs = []
        else:
            raise TypeError("configs should be a list of Configs or None")

        if time is not None:
            self.define_time(time)
        self.run_config = run_config

    def add_config(self, config: Config):
        self.configs.append(config)
        _verify_config_consistency(self.configs)

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)
        _verify_scenario_consistency(self.scenarios)

    def define_time(self, time: np.ndarray):
        if not hasattr(time, '__iter__'):
            raise TimeNotIterableError("Time must be an iterable data type")
        self.time = np.asarray(time)

    def remove_config(self, config: Config):
        self.configs.remove(config)

    def remove_scenario(self, scenario: Scenario):
        self.scenarios.remove(scenario)

    def _assign_indices(self):
        # Now that we know that scenarios and configs are consistent, we can
        # allocate array indices to them for running the model. We also define
        # a class level attribute "species".
        self.scenarios_index_mapping = {}
        self.species_index_mapping = {}
        self.category_indices = []
        self.configs_index_mapping = {}
        self.ghg_indices = []
        self.slcf_indices = []
        self.ghg_emissions_indices = []
        self.ghg_concentration_indices = []
        self.minor_ghg_indices = []
        self.halogen_indices = []
        self.ari_index = None
        self.lapsi_index = None
        self.h2o_stratospheric_index = None
        self.nox_aviation_index = None
        self.aci_index = None
        self.ozone_index = None
        self.contrails_index = None
        self.land_use_index = None
        self.co2_ffi_index = None
        self.co2_afolu_index = None
        self.co2_index = None
        self.ch4_index = None
        self.n2o_index = None
        self.cfc11_index = None
        self.bc_index = None
        self.so2_index = None
        self.oc_index = None

        for ispec, specie in enumerate(self.species):  # specie is a SpeciesID
            self.species_index_mapping[specie.name] = ispec
            self.category_indices.append(specie.category)
            if specie.category in AggregatedCategory.GREENHOUSE_GAS:
                self.ghg_indices.append(ispec)
                if specie.run_mode in (RunMode.EMISSIONS, RunMode.FROM_OTHER_SPECIES):
                    self.ghg_emissions_indices.append(ispec)
                elif specie.run_mode == RunMode.CONCENTRATION:
                    self.ghg_concentration_indices.append(ispec)
            if specie.category in AggregatedCategory.SLCF:
                self.slcf_indices.append(ispec)
            if specie.category in AggregatedCategory.MINOR_GREENHOUSE_GAS:
                self.minor_ghg_indices.append(ispec)
            if specie.category in AggregatedCategory.HALOGEN:
                self.halogen_indices.append(ispec)
            if specie.category == Category.AEROSOL_RADIATION_INTERACTIONS:
                self.ari_index = ispec
            if specie.category == Category.LAPSI:
                self.lapsi_index = ispec
            if specie.category == Category.H2O_STRATOSPHERIC:
                self.h2o_stratospheric_index = ispec
            if specie.category == Category.NOX_AVIATION:
                self.nox_aviation_index = ispec
            if specie.category == Category.AEROSOL_CLOUD_INTERACTIONS:
                self.aci_index = ispec
            if specie.category == Category.OZONE:
                self.ozone_index = ispec
            if specie.category == Category.CONTRAILS:
                self.contrails_index = ispec
            if specie.category == Category.LAND_USE:
                self.land_use_index = ispec
            if specie.category == Category.CO2_FFI:
                self.co2_ffi_index = ispec
            if specie.category == Category.CO2_AFOLU:
                self.co2_afolu_index = ispec
            if specie.category == Category.CO2:
                self.co2_index = ispec
            if specie.category == Category.CH4:
                self.ch4_index = ispec
            if specie.category == Category.N2O:
                self.n2o_index = ispec
            if specie.category == Category.CFC_11:
                self.cfc_11_index = ispec
            if specie.category == Category.SULFUR:
                self.so2_index = ispec
            if specie.category == Category.BC:
                self.bc_index = ispec
            if specie.category == Category.OC:
                self.oc_index = ispec
            if specie.category == Category.SOLAR:
                self.solar_index = ispec
            if specie.category == Category.VOLCANIC:
                self.volcanic_index = ispec
        for iscen, scenario in enumerate(self.scenarios):
            self.scenarios_index_mapping[scenario.name] = iscen
        for iconf, config in enumerate(self.configs):
            self.configs_index_mapping[config.name] = iconf


    # TODO: incomplete
    def _fill_concentration(self):
        """After the emissions to concentrations step we want to put the concs into each GreenhouseGas"""
        for ispec, specie in enumerate(self.species_index_mapping):
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
        self.erfari_radiative_efficiency_array = np.zeros((1, 1, n_configs, n_species, 1)) * np.nan
        self.erfaci_scale_array = np.ones((1, 1, n_configs, 1, 1)) * np.nan
        self.erfaci_shape_sulfur_array = np.ones((1, 1, n_configs, 1, 1)) * np.nan
        self.erfaci_shape_bcoc_array = np.ones((1, 1, n_configs, 1, 1)) * np.inf
        self.ozone_radiative_efficiency_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.contrails_radiative_efficiency_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.lapsi_emissions_to_forcing_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.stratospheric_h2o_factor_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.land_use_cumulative_emissions_to_forcing_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        # TODO: make a more general temperature-forcing feedback for all species
        self.forcing_temperature_feedback_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        # TODO: start from non-zero temperature
        self.temperature = np.ones((n_timesteps, n_scenarios, n_configs, 1, self.run_config.n_temperature_boxes)) * np.nan
        self.toa_imbalance = np.ones((n_timesteps, n_scenarios, n_configs, 1, 1)) * np.nan

        for ispec, specie in enumerate(self.species_index_mapping):
            for iconf, config in enumerate(self.configs_index_mapping):
                conf_spec = self.configs[iconf].species_configs[ispec]
                self.forcing_scaling_array[:, 0, iconf, ispec, 0] = (1+conf_spec.tropospheric_adjustment) * conf_spec.scale
                self.efficacy_array[:, 0, iconf, ispec, 0] = conf_spec.efficacy
                self.baseline_emissions_array[:, :, iconf, ispec, :] = conf_spec.baseline_emissions
                self.forcing_temperature_feedback_array[:, :, iconf, ispec, :] = conf_spec.forcing_temperature_feedback
                self.lapsi_emissions_to_forcing_array[:, :, iconf, ispec, :] = conf_spec.lapsi_emissions_to_forcing
                self.stratospheric_h2o_factor_array[0, 0, iconf, ispec, 0] = conf_spec.h2o_stratospheric_factor
                self.land_use_cumulative_emissions_to_forcing_array[0, 0, iconf, ispec, 0] = conf_spec.land_use_cumulative_emissions_to_forcing
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
                self.erfari_radiative_efficiency_array[:, 0, iconf, ispec, :] = conf_spec.erfari_radiative_efficiency
                if self.species[ispec].category == Category.AEROSOL_CLOUD_INTERACTIONS:
                    self.erfaci_scale_array[0, 0, iconf, 0, 0] = conf_spec.aci_params['scale']
                    self.erfaci_shape_sulfur_array[0, 0, iconf, 0, 0] = conf_spec.aci_params['Sulfur']
                    if aci_method==ACIMethod.SMITH2018:
                        self.erfaci_shape_bcoc_array[0, 0, iconf, 0, 0] = conf_spec.aci_params['BC+OC']
                self.ozone_radiative_efficiency_array[0, 0, iconf, ispec, 0] = conf_spec.ozone_radiative_efficiency
                self.contrails_radiative_efficiency_array[0, 0, iconf, ispec, 0] = conf_spec.contrails_radiative_efficiency
            for iscen, scenario_name in enumerate(self.scenarios_index_mapping):
                scen_spec = self.scenarios[iscen].list_of_species[ispec]
                if self.species[ispec].run_mode == RunMode.EMISSIONS:
                    self.emissions_array[:, iscen, :, ispec, 0] = scen_spec.emissions[:, None]
                if self.species[ispec].run_mode == RunMode.CONCENTRATION:
                    self.concentration_array[:, iscen, :, ispec, 0] = scen_spec.concentration[:, None]
                if self.species[ispec].run_mode == RunMode.FORCING:
                    self.forcing_array[:, iscen, :, ispec, 0] = scen_spec.forcing[:, None] * self.forcing_scaling_array[0, 0, :, ispec, 0]

        # add aggregated CO2 emissions
        if self.co2_ffi_index is not None and self.co2_afolu_index is not None:
            self.emissions_array[:, :, :, self.co2_index, :] = (
                self.emissions_array[:, :, :, self.co2_ffi_index, :] +
                self.emissions_array[:, :, :, self.co2_afolu_index, :]
            )

        self.cumulative_emissions_array = np.cumsum(self.emissions_array * self.time_deltas[:, None, None, None, None], axis=TIME_AXIS)
        self.alpha_lifetime_array = np.ones((n_timesteps, n_scenarios, n_configs, n_species, 1))
        self.airborne_emissions_array = np.zeros((n_timesteps, n_scenarios, n_configs, n_species, 1))
        self.stochastic_forcing = np.ones((n_timesteps, n_scenarios, n_configs)) * np.nan

        # concentration-driven GHG emissions and cumulative emissions are calculated dynamically and initialised to be zero
        self.emissions_array[:,:,:,self.ghg_concentration_indices,:] = 0
        self.cumulative_emissions_array[:,:,:,self.ghg_concentration_indices,:] = 0

    def _pre_run_checks(self):
        # Check if necessary inputs are defined
        for attr in ('scenarios', 'configs', 'time'):
            if not hasattr(self, attr):
                raise MissingInputError(
                    f"{attr} was not provided when trying to run"
                )
        _check_type_of_elements(self.scenarios, Scenario, 'scenarios')
        _check_type_of_elements(self.configs, Config, 'configs')
        self.species = _map_species_scenario_config(self.scenarios, self.configs, self.run_config)
        if self.n_timesteps != len(self.time):
            raise TimeMismatchError(
                f"time vector provided is of length {len(self.time)} whereas "
                f"the supplied Scenario inputs are of length "
                f"{self.n_timesteps}."
            )

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
        alpha_lifetime_array = np.ones((1, n_scenarios, n_configs, n_species, 1))

        # initialise the energy balance model and get critical vectors
        # which itself needs to be run once per "config" and dimensioned correctly
        eb_matrix_d, forcing_vector_d, stochastic_d = _make_ebm(self.configs, n_timesteps)

        # Main loop
        for i_timestep in range(n_timesteps):

            # 1. calculate scaling of atmospheric lifetimes (alpha)
            # concentration-driven GHGs need cumulative emissions updating each timestep
            self.cumulative_emissions_array[i_timestep, :, :, self.ghg_concentration_indices, :] = (
                self.cumulative_emissions_array[i_timestep-1, :, :, self.ghg_concentration_indices, :] +
                self.emissions_array[i_timestep-1, :, :, self.ghg_concentration_indices, :]
            )
            # A quirk of numpy requires dropping the last dimension here.
            alpha_lifetime_array[0:1, :, :, self.ghg_indices] = calculate_alpha(
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
            )[0:1, :, :, self.ghg_indices, :]
            self.alpha_lifetime_array[[i_timestep], ...] = alpha_lifetime_array

            # 2. GHG emissions to concentrations
            ae_timestep = i_timestep-1 if i_timestep>0 else 0
            (
                self.concentration_array[i_timestep:i_timestep+1, :, :, self.ghg_emissions_indices, :],
                gas_boxes[0:1, :, :, self.ghg_emissions_indices, :],
                self.airborne_emissions_array[i_timestep:i_timestep+1, :, :, self.ghg_emissions_indices, :]
            ) = step_concentration(
                self.emissions_array[i_timestep:i_timestep+1, :, :, self.ghg_emissions_indices, :],
                gas_boxes[0:1, :, :, self.ghg_emissions_indices, :],
                self.airborne_emissions_array[ae_timestep:ae_timestep+1, :, :, self.ghg_emissions_indices, :],
                self.burden_per_emission_array[0:1, :, :, self.ghg_emissions_indices, :],
                self.lifetime_array[0:1, :, :, self.ghg_emissions_indices, :],
                alpha_lifetime=alpha_lifetime_array[0:1, :, :, self.ghg_emissions_indices, :],
                pre_industrial_concentration=self.baseline_concentration_array[0:1, :, :, self.ghg_emissions_indices, :],
                timestep=self.time_deltas[i_timestep],
                partition_fraction=self.partition_fraction_array[0:1, :, :, self.ghg_emissions_indices, :],
                natural_emissions_adjustment=self.natural_emissions_adjustment_array[0:1, :, :, self.ghg_emissions_indices, :],
            )

            # 3. GHG concentrations to emissions:
            (
                self.emissions_array[i_timestep:i_timestep+1, :, :, self.ghg_concentration_indices, :],
                gas_boxes[0:1, :, :, self.ghg_concentration_indices, :],
                self.airborne_emissions_array[i_timestep:i_timestep+1, :, :, self.ghg_concentration_indices, :]
            ) = unstep_concentration(
                self.concentration_array[i_timestep:i_timestep+1, :, :, self.ghg_concentration_indices, :],
                gas_boxes[0:1, :, :, self.ghg_concentration_indices, :],
                self.airborne_emissions_array[ae_timestep:ae_timestep+1, :, :, self.ghg_concentration_indices, :],
                self.burden_per_emission_array[0:1, :, :, self.ghg_concentration_indices, :],
                self.lifetime_array[0:1, :, :, self.ghg_concentration_indices, :],
                alpha_lifetime=alpha_lifetime_array[0:1, :, :, self.ghg_concentration_indices, :],
                pre_industrial_concentration=self.baseline_concentration_array[0:1, :, :, self.ghg_concentration_indices, :],
                timestep=self.time_deltas[i_timestep],
                partition_fraction=self.partition_fraction_array[0:1, :, :, self.ghg_concentration_indices, :],
                natural_emissions_adjustment=self.natural_emissions_adjustment_array[0:1, :, :, self.ghg_concentration_indices, :],
            )

            # 4. greenhouse gas concentrations to forcing
            self.forcing_array[i_timestep:i_timestep+1, :, :, self.ghg_indices, :] = calculate_ghg_forcing(
                self.concentration_array[i_timestep:i_timestep+1, ...],
                self.baseline_concentration_array,
                self.forcing_scaling_array,
                self.radiative_efficiency_array,
                self.co2_index, self.ch4_index, self.n2o_index, self.minor_ghg_indices
            )[0:1, :, :, self.ghg_indices, :]

            # 5. aerosol direct emissions to forcing
            if self.ari_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, self.ari_index, :] = calculate_erfari_forcing(
                    self.emissions_array[[i_timestep], ...],
                    self.concentration_array[[i_timestep], ...],
                    self.baseline_emissions_array,
                    self.baseline_concentration_array,
                    self.forcing_scaling_array,
                    self.erfari_radiative_efficiency_array,
                    self.slcf_indices,
                    self.ghg_indices,
                )

            # 6. aerosol indirect emissions to forcing
            if self.aci_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, self.aci_index, :] = calculate_erfaci_forcing(
                    self.emissions_array[[i_timestep], ...],
                    self.baseline_emissions_array,
                    self.forcing_scaling_array,
                    self.erfaci_scale_array,
                    self.erfaci_shape_sulfur_array,
                    self.erfaci_shape_bcoc_array,
                    self.so2_index,
                    self.bc_index,
                    self.oc_index,
                    self.run_config.aci_method
                )[0:1, :, :, self.aci_index, :]

            # 7. ozone emissions and concentrations to forcing
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
                    self.cfc_11_index,
                    self.halogen_indices,
                    self.slcf_indices,
                    self.ghg_indices,
                )

            # 8. contrail emissions from aviation NOx to forcing
            if self.contrails_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, [self.contrails_index], :] = calculate_linear_forcing(
                    self.emissions_array[[i_timestep], ...],
                    self.baseline_emissions_array,
                    self.forcing_scaling_array,
                    self.contrails_radiative_efficiency_array,
                    [self.nox_aviation_index]
                )

            # 9. BC emissions to LAPSI forcing
            # TODO: can be refactored with aerosol direct I think
            # TODO: include OC
            if self.lapsi_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, [self.lapsi_index], :] = calculate_linear_forcing(
                    self.emissions_array[[i_timestep], ...],
                    self.baseline_emissions_array,
                    self.forcing_scaling_array,
                    self.lapsi_emissions_to_forcing_array,
                )

            # 10. CH4 forcing to stratospheric water vapour forcing
            if self.h2o_stratospheric_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, [self.h2o_stratospheric_index], :] = calculate_linear_forcing(
                    self.forcing_array[[i_timestep], ...],
                    self.forcing_array[0:1, ...],
                    self.forcing_scaling_array[:, :, :, [self.h2o_stratospheric_index], :],
                    self.stratospheric_h2o_factor_array,
                )

            # 11. CO2 cumulative emissions to land use change forcing
            # TODO: separate treatment depending on how land use is formulated
            if self.land_use_index is not None:
                self.forcing_array[i_timestep:i_timestep+1, :, :, [self.land_use_index], :] = calculate_linear_forcing(
                    self.cumulative_emissions_array[[i_timestep], ...],
                    self.cumulative_emissions_array[0:1, ...],
                    self.forcing_scaling_array[:, :, :, [self.land_use_index], :],
                    self.land_use_cumulative_emissions_to_forcing_array,
                )

            # 12. In future we should allow volcanic forcing to have a temperature dependence

            # 13. sum up all of the forcing calculated previously
            self.forcing_sum_array[[i_timestep], ...] = np.nansum(
                self.forcing_array[[i_timestep], ...], axis=SPECIES_AXIS, keepdims=True
            )
            efficacy_adjusted_forcing=np.nansum(
                self.forcing_array[[i_timestep], ...]*self.efficacy_array, axis=SPECIES_AXIS, keepdims=True
            )

            # 14. run the energy balance model
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
                    self.toa_imbalance[i_timestep, iscen, iconf, :, :] = (
                        self.forcing_sum_array[i_timestep, iscen, iconf, :, :] -
                        config.climate_response.ocean_heat_transfer[0]*
                        self.temperature[i_timestep, iscen, iconf, :, 0] +
                        (1 - config.climate_response.deep_ocean_efficacy) * config.climate_response.ocean_heat_transfer[2]
                        * (self.temperature[i_timestep, iscen, iconf, :, 1] -
                         self.temperature[i_timestep, iscen, iconf, :, 2])
                    )
            # TODO: fill in OHC
        self._fill_concentration()
        self._fill_forcing()
        self._fill_temperature()
