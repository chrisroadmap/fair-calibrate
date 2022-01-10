from abc import ABC
from collections.abc import Iterable
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from numbers import Number
import typing

import numpy as np

from energy_balance_model import EnergyBalanceModel
from exceptions import (
    DuplicationError,
    IncompatibleConfigError,
    MissingInputError,
    SpeciesMismatchError,
    TimeMismatchError,
    UnexpectedInputError,
    WrongArrayShapeError
)

# each Scenario will contain a list of Species
# each Config will contain settings, as well as options relating to each Species
# FAIR can contain one or more Scenarios and one or more Configs

IIRF_HORIZON = 100
iirf_max = 99.95
M_ATMOS = 5.1352e18
MOLWT_AIR = 28.97
TIME_AXIS = 0
SPECIES_AXIS = 3
GAS_BOX_AXIS = 4

# at top level
#class CH4(SpeciesID):
#   greenhouse_gas: True
#   ozone_precursor: True
#   ozone_method: 'concentration'
#   aerosol_direct_precursor: False
#   aerosol_indirect_precursor: False
#   co2_precursor: True
#   stratospheric_water_vapour_precursor: True
#   contrail_precursor: False
#   lapsi_precursor: False
#   landuse_precursor: False


class Category(Enum):
    """Types of Species encountered in climate scenarios."""
    CO2_FFI = auto()
    CO2_AFOLU = auto()
    CO2 = auto()
    CH4 = auto()
    N2O = auto()
    CFC_11 = auto()
    OTHER_HALOGEN = auto()
    F_GAS = auto()
    SULFUR = auto()
    BC = auto()
    OC = auto()
    OTHER_AEROSOL = auto()
    OZONE_PRECURSOR = auto()
    OZONE = auto()
    AEROSOL_CLOUD_INTERACTIONS = auto()
    CONTRAILS = auto()
    LAPSI = auto()
    LAND_USE = auto()
    VOLCANIC = auto()
    SOLAR = auto()

GREENHOUSE_GAS = [Category.CO2_FFI, Category.CO2_AFOLU, Category.CO2, Category.CH4, Category.N2O, Category.CFC_11, Category.OTHER_HALOGEN, Category.F_GAS]
HALOGEN = [Category.CFC_11, Category.OTHER_HALOGEN]
AEROSOL = [Category.SULFUR, Category.BC, Category.OC, Category.OTHER_AEROSOL]
NO_DUPLICATES_ALLOWED = [
    Category.CO2,
    Category.CH4,
    Category.N2O,
    Category.CFC_11,
    Category.SULFUR,
    Category.BC,
    Category.OC,
    Category.OZONE,
    Category.AEROSOL_CLOUD_INTERACTIONS,
    Category.CONTRAILS,
    Category.LAPSI,
    Category.LAND_USE,
    Category.SOLAR,
    Category.VOLCANIC
]


class RunMode(Enum):
    EMISSIONS = auto()
    CONCENTRATION = auto()
    FROM_OTHER_SPECIES = auto()
    FORCING = auto()


valid_run_modes = {
    Category.CO2_FFI: (RunMode.EMISSIONS,),
    Category.CO2_AFOLU: (RunMode.EMISSIONS,),
    Category.CO2: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.CH4: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.N2O: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.CFC_11: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.OTHER_HALOGEN: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.F_GAS: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.SULFUR: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.BC: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.OC: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.OTHER_AEROSOL: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.OZONE_PRECURSOR: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.OZONE: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.AEROSOL_CLOUD_INTERACTIONS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.CONTRAILS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.LAPSI: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.LAND_USE: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.SOLAR: (RunMode.FORCING,),
    Category.VOLCANIC: (RunMode.FORCING,),
}

# top level
@dataclass
class SpeciesID():
    name: str
    category: Category
    run_mode: RunMode=None

    def __post_init__(self):
        # 1. fill default run_mode
        if self.run_mode is None:
            if self.category in [Category.SOLAR, Category.VOLCANIC]:
                self.run_mode = RunMode.FORCING
            elif self.category in [Category.OZONE, Category.AEROSOL_CLOUD_INTERACTIONS, Category.CONTRAILS, Category.LAPSI, Category.LAND_USE]:
                self.run_mode = RunMode.FROM_OTHER_SPECIES
            else:
                self.run_mode = RunMode.EMISSIONS
        # 2. check valid run mode for each species given
        if self.run_mode not in valid_run_modes[self.category]:
            raise InvalidRunModeError(f"cannot run {self.category} in {self.run_mode} mode")


# scenario level
@dataclass
class Species():
    species_id: SpeciesID
    emissions: np.ndarray=None
    concentration: np.ndarray=None
    forcing: np.ndarray=None
    #run_mode: field() - define at top level

    def __post_init__(self):
        # 1. Input validation
        if self.species_id.run_mode == RunMode.EMISSIONS and self.emissions is None:
            raise MissingInputError(f"for {self.species_id.name} run in emissions mode, emissions must be specified")
        if self.species_id.run_mode == RunMode.CONCENTRATION and self.concentration is None:
            raise MissingInputError(f"for {self.species_id.name} run in concentration mode, concentration must be specified")
        if self.species_id.run_mode == RunMode.FORCING and self.forcing is None:
            raise MissingInputError(f"for {self.species_id.name} run in forcing mode, forcing must be specified")


class AciMethod(Enum):
    STEVENS2015 = auto()
    SMITH2018 = auto()

# top level: consider excluding "Config"
@dataclass
class RunConfig():
    n_gas_boxes = 4
    n_temperature_boxes = 3
    temperature_prescribed = False
    aci_method = AciMethod.SMITH2018

# top level?

# config level
@dataclass
class SpeciesConfig():
    species_id: SpeciesID
    molecular_weight: float=None
    lifetime: np.ndarray=None
    partition_fraction: np.ndarray=1
    radiative_efficiency: float=None
    iirf_0: float=field(default=None)
    iirf_airborne: float=0
    iirf_cumulative: float=0
    iirf_temperature: float=0
    natural_emissions_adjustment: float=0
    baseline_concentration: float=0
    erfari_emissions_to_forcing: float=0
    lapsi_emissions_to_forcing: float=0
    baseline_emissions: float=0
    ozone_radiative_efficiency: float=None
    cl_atoms: float=None
    br_atoms: float=None
    fractional_release: float=None
    tropospheric_adjustment: float=0
    scale: float=1
    efficacy: float=1

    def __post_init__(self):
        # validate input - the whole partition_fraction and lifetime thing
        # would be nice to validate if not CO2, CH4 or N2O that radiative_efficiency must be defined.
        if self.species_id.category in GREENHOUSE_GAS:
            self.g1 = np.sum(
                np.asarray(self.partition_fraction) * np.asarray(self.lifetime) *
                (1 - (1 + IIRF_HORIZON/np.asarray(self.lifetime)) *
                np.exp(-IIRF_HORIZON/np.asarray(self.lifetime)))
            )
            self.g0 = np.exp(-1 * np.sum(np.asarray(self.partition_fraction)*
                np.asarray(self.lifetime)*
                (1 - np.exp(-IIRF_HORIZON/np.asarray(self.lifetime))))/
                self.g1
            )
            self.burden_per_emission = 1 / (M_ATMOS / 1e18 * self.molecular_weight / MOLWT_AIR)
            if self.iirf_0 is None:
                self.iirf_0 = (
                    np.sum(np.asarray(self.lifetime) *
                    (1 - np.exp(-IIRF_HORIZON / np.asarray(self.lifetime)))
                    * np.asarray(self.partition_fraction))
                )

        if self.species_id.category in HALOGEN:
            if ~isinstance(self.ozone_radiative_efficiency, Number):
                raise ValueError("ozone_properties.ozone_radiative_efficiency should be a number for Halogens")
            if ~isinstance(self.cl_atoms, int) or self.cl_atoms < 0:
                raise ValueError("ozone_properties.cl_atoms should be a non-negative integer for Halogens")
            if ~isinstance(self.br_atoms, int) or self.cl_atoms < 0:
                raise ValueError("ozone_properties.br_atoms should be a non-negative integer for Halogens")
            if ~isinstance(self.fractional_release, Number) or self.fractional_release < 0:
                raise ValueError("ozone_properties.fractional_release should be a non-negative number for Halogens")

        if self.species_id.category == Category.AEROSOL_CLOUD_INTERACTIONS:
            pass
            #TODO: CHECK whether correct input is provided.


@dataclass
class Scenario():
    name: str
    list_of_species: typing.List[Species]

    def __post_init__(self):
        if not isinstance(self.list_of_species, list):
            raise TypeError('list_of_species argument passed to Scenario must be a list of Species')

        # check for Categories for which it makes no sense to duplicate
        running_total = {category: 0 for category in NO_DUPLICATES_ALLOWED}
        major_ghgs_forward_mode = 0
        for species in self.list_of_species:
            if species.species_id.category in NO_DUPLICATES_ALLOWED:
                running_total[species.species_id.category] = running_total[species.species_id.category] + 1
                if running_total[species.species_id.category] > 1:
                    raise DuplicationError(
                        f"The Scenario contains more than one instance of "
                        f"{species.species_id.category}. This is not valid."
                    )
            if species.species_id.category in [Category.CO2, Category.CH4, Category.N2O]:
                if species.species_id.run_mode in [RunMode.EMISSIONS, RunMode.CONCENTRATION]:
                    major_ghgs_forward_mode = major_ghgs_forward_mode + 1
        n_major_ghgs = running_total[Category.CO2] + running_total[Category.CH4] + running_total[Category.N2O]
        if 0 < n_major_ghgs < 3:
            if major_ghgs_forward_mode > 0: #TODO and emissions or concentration driven mode
                raise IncompatibleConfigError(
                    f"Either all of CO2, CH4 and N2O must be given in a Scenario, or "
                    f"none, unless those provided are forcing-driven. If you want to "
                    f"exclude the effect of one or two of these gases, consider setting "
                    f"emissions of these gases to zero or concentrations to pre-industrial."
                )

@dataclass
class ClimateResponse():
    ocean_heat_capacity: typing.Union[Iterable, float]
    ocean_heat_transfer: typing.Union[Iterable, float]
    deep_ocean_efficacy: float=1
    stochastic_run: bool=False
    sigma_xi: float=0.5
    sigma_eta: float=0.5
    gamma_autocorrelation: float=2.0
    seed: int=None

    def __post_init__(self):
        self.ocean_heat_capacity=np.asarray(self.ocean_heat_capacity)
        self.ocean_heat_transfer=np.asarray(self.ocean_heat_transfer)
        if self.ocean_heat_capacity.ndim != 1 or self.ocean_heat_transfer.ndim != 1:
            raise WrongArrayShapeError(
                f"In ClimateResponse, both the ocean_heat_capacity and "
                f"ocean_heat_transfer arguments must be castable to 1-D arrays."
            )
        if len(self.ocean_heat_capacity) != len(self.ocean_heat_transfer):
            raise IncompatibleConfigError(
                f"In ClimateResponse, the length of the ocean_heat_capacity "
                f"({len(self.ocean_heat_capacity)}) differs from the length of "
                f"ocean_heat_transfer ({len(self.ocean_heat_transfer)})"
            )
        if not isinstance(self.deep_ocean_efficacy, Number) or self.deep_ocean_efficacy<=0:
            raise TypeError("deep_ocean_efficacy must be a positive number")
        if self.stochastic_run:
            for attr in ('sigma_xi', 'sigma_eta', 'gamma_autocorrelation'):
                if not isinstance(getattr(self, attr), Number) or getattr(self, attr)<=0:
                    raise TypeError(
                        f"If stochastic_run is True, then {attr} must be a "
                        f"positive number."
                    )


@dataclass
class Config():
    name: str
    climate_response: ClimateResponse
    species_configs: typing.List[SpeciesConfig]
    aci_params: dict=None

    def __post_init__(self):
        # check eveything provided is a Config
        if not isinstance(self.species_configs, list):
            raise TypeError('species_configs argument passed to Config must be a list of SpeciesConfig')
        # fill aci_params with AR6 defaults if not provided.
        if self.aci_params is None:
            self.aci_params={"scale": 2.09841432, "Sulfur": 260.34644166, "BC+OC": 111.05064063}

# TODO: radiative efficiency for the big three should be calculated internally
default_species_config = {
    'co2' : SpeciesConfig(
        species_id = SpeciesID(name='CO2', category=Category.CO2, run_mode=RunMode.EMISSIONS),
        molecular_weight = 44.009,
        lifetime = np.array([1e9, 394.4, 36.54, 4.304]),
        partition_fraction = np.array([0.2173, 0.2240, 0.2824, 0.2763]),
        radiative_efficiency = 1.3344985680386619e-05,
        iirf_0=29,
        iirf_airborne=0.000819,
        iirf_cumulative=0.00846,
        iirf_temperature=4.0,
        baseline_concentration = 278.3,
        tropospheric_adjustment = 0.05
    ),
    'ch4' : SpeciesConfig(
        species_id = SpeciesID('CH4', Category.CH4, run_mode=RunMode.EMISSIONS),
        molecular_weight = 16.043,
        lifetime = 8.25,
        radiative_efficiency = 0.00038864402860869495,
        iirf_airborne = 0.00032,
        iirf_temperature = -0.3,
        baseline_concentration = 729.2,
        tropospheric_adjustment = -0.14,
        ozone_radiative_efficiency = 1.75e-4,
    ),
    'n2o': SpeciesConfig(
        species_id = SpeciesID('N2O', Category.N2O, run_mode=RunMode.EMISSIONS),
        molecular_weight = 44.013,
        lifetime = 109,
        radiative_efficiency = 0.00319550741640458,
        baseline_concentration = 270.1,
        tropospheric_adjustment = 0.07,
        ozone_radiative_efficiency = 7.1e-4,

    ),
    'sulfur': SpeciesConfig(
        species_id = SpeciesID('Sulfur', Category.SULFUR, run_mode=RunMode.EMISSIONS),
        erfari_emissions_to_forcing = -0.0036167830509091486,
        baseline_emissions = 2.44004843482201
    ),
    'bc': SpeciesConfig(
        species_id = SpeciesID('BC', Category.BC, run_mode=RunMode.EMISSIONS),
        erfari_emissions_to_forcing = 0.0507748226795483,
        baseline_emissions = 2.09777075542297,
    ),
    'oc': SpeciesConfig(
        species_id = SpeciesID('OC', Category.OC, run_mode=RunMode.EMISSIONS),
        erfari_emissions_to_forcing = -0.006214374446217472,
        baseline_emissions = 15.4476681469614,
    ),
    'nh3': SpeciesConfig(
        species_id = SpeciesID('NH3', Category.OTHER_AEROSOL, run_mode=RunMode.EMISSIONS),
        erfari_emissions_to_forcing = -0.0020809236231100624,
        baseline_emissions = 6.92769009144426
    ),
    'co': SpeciesConfig(
        species_id = SpeciesID('CO', Category.OZONE_PRECURSOR),
#        ozone_radiative_efficiency =,
        baseline_emissions = 348.52735877736
    ),
    'nox' : SpeciesConfig(
        species_id = SpeciesID('NOx', Category.OZONE_PRECURSOR),
#        ozone_radiative_efficiency =,
        baseline_emissions = 12.7352119423177
    ),
    'voc' : SpeciesConfig(
        species_id = SpeciesID('VOC', Category.OZONE_PRECURSOR),
#        ozone_radiative_efficiency =,
        baseline_emissions = 60.0218262241548
    ),
}

def species_config_from_default(name, **kwargs):
    config = default_species_config[name.lower()]
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


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


def check_included(aci_params, aci_method):
    required_params = {
        AciMethod.SMITH2018: ['scale', 'Sulfur', 'BC+OC'],
        AciMethod.STEVENS2015: ['scale', 'Sulfur']
    }
    if list(aci_params) != required_params[aci_method]:
        raise IncompatibleConfigError(
            f"For aerosol-cloud interactions using the {run_config.aci_method}, "
            f"the aci_params in the construction of Config must include "
            f"{required_params[aci_method]}."
        )



def map_species_scenario_config(scenarios, configs, run_config):
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
        AciMethod.SMITH2018: [Category.SULFUR, Category.BC, Category.OC],
        AciMethod.STEVENS2015: [Category.SULFUR],
    }
    aerosol_species_included = []
    aci_desired = False
    for ispec, species in enumerate(scenarios[0].list_of_species):
        species_included_first_scenario.append(scenarios[0].list_of_species[ispec].species_id)
        if scenarios[0].list_of_species[ispec].species_id.category in required_aerosol_species[run_config.aci_method] and scenarios[0].list_of_species[ispec].species_id.run_mode == RunMode.EMISSIONS:
            aerosol_species_included.append(scenarios[0].list_of_species[ispec].species_id.category)
        if scenarios[0].list_of_species[ispec].species_id.category == Category.AEROSOL_CLOUD_INTERACTIONS:
            aci_desired = True
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
        for config in configs:
            check_included(config.aci_params, run_config.aci_method)
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


def calculate_alpha(
    cumulative_emissions,
    airborne_emissions,
    temperature,
    iirf_0,
    iirf_cumulative,
    iirf_temperature,
    iirf_airborne,
    g0,
    g1,
    iirf_max = iirf_max,
):
    """
    Calculate greenhouse-gas time constant scaling factor.

    Parameters
    ----------
    cumulative_emissions : ndarray
        GtC cumulative emissions since pre-industrial.
    airborne_emissions : ndarray
        GtC total emissions remaining in the atmosphere.
    temperature : ndarray or float
        K temperature anomaly since pre-industrial.
    iirf_0 : ndarray
        pre-industrial time-integrated airborne fraction.
    iirf_cumulative : ndarray
        sensitivity of time-integrated airborne fraction with atmospheric
        carbon stock.
    iirf_temperature : ndarray
        sensitivity of time-integrated airborne fraction with temperature
        anomaly.
    iirf_airborne : ndarray
        sensitivity of time-integrated airborne fraction with airborne
        emissions.
    g0 : ndarray
        parameter for alpha TODO: description
    g1 : ndarray
        parameter for alpha TODO: description
    iirf_max : float
        maximum allowable value to time-integrated airborne fraction

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    alpha : float
        scaling factor for lifetimes
    """

    iirf = iirf_0 + iirf_cumulative * (cumulative_emissions-airborne_emissions) + iirf_temperature * temperature + iirf_airborne * airborne_emissions
    iirf = (iirf>iirf_max) * iirf_max + iirf * (iirf<iirf_max)
    alpha = g0 * np.exp(iirf / g1)

    return alpha

def step_concentration(
    emissions,
    gas_boxes_old,
    airborne_emissions_old,
    burden_per_emission,
    lifetime,
    alpha_lifetime,
    partition_fraction,
    pre_industrial_concentration,
    timestep=1,
    natural_emissions_adjustment=0,
):
    """
    Calculates concentrations from emissions of any greenhouse gas.

    Parameters
    ----------
    emissions : ndarray
        emissions rate (emissions unit per year) in timestep.
    gas_boxes_old : ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the previous timestep.
    airborne_emissions_old : ndarray
        The total airborne emissions at the beginning of the timestep. This is
        the concentrations above the pre-industrial control. It is also the sum
        of gas_boxes_old if this is an array.
    burden_per_emission : ndarray
        how much atmospheric concentrations grow (e.g. in ppm) per unit (e.g.
        GtCO2) emission.
    lifetime : ndarray
        atmospheric burden lifetime of greenhouse gas (yr). For multiple
        lifetimes gases, it is the lifetime of each box.
    alpha_lifetime : ndarray
        scaling factor for `lifetime`. Necessary where there is a state-
        dependent feedback.
    partition_fraction : ndarray
        the partition fraction of emissions into each gas box. If array, the
        entries should be individually non-negative and sum to one.
    pre_industrial_concentration : ndarray
        pre-industrial concentration of gas(es) in question.
    timestep : float, default=1
        emissions timestep in years.
    natural_emissions_adjustment : ndarray or float, default=0
        Amount to adjust emissions by for natural emissions given in the total
        in emissions files.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
    on the same time intervals: the airborne_emissions values are on time
    boundaries and these are averaged before being returned.

    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    concentration_out : ndarray
        greenhouse gas concentrations at the centre of the timestep.
    gas_boxes_new : ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the timestep.
    airborne_emissions_new : ndarray
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """

    decay_rate = timestep/(alpha_lifetime * lifetime)
    decay_factor = np.exp(-decay_rate)

    gas_boxes_new = (
        partition_fraction *
        (emissions-natural_emissions_adjustment) *
        1 / decay_rate *
        (1 - decay_factor) * timestep + gas_boxes_old * decay_factor
    )
    airborne_emissions_new = np.sum(gas_boxes_new, axis=GAS_BOX_AXIS, keepdims=True)
    concentration_out = (
        pre_industrial_concentration +
        burden_per_emission * (
            airborne_emissions_new + airborne_emissions_old
        ) / 2
    )

    return concentration_out, gas_boxes_new, airborne_emissions_new


def calculate_ghg_forcing(
    concentration,
    pre_industrial_concentration,
    tropospheric_adjustment,
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
    tropospheric_adjustment : ndarray
        conversion factor from radiative forcing to effective radiative forcing.
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
    erf_out[:, :, :, [gas_index_mapping["CO2"]], :] = (alpha_p + alpha_n2o) * np.log(co2/co2_pi) * (1 + tropospheric_adjustment[:, :, :, [gas_index_mapping["CO2"]], :])

    # CH4
    erf_out[:, :, :, [gas_index_mapping["CH4"]], :] = (
        (a3*np.sqrt(ch4) + b3*np.sqrt(n2o) + d3) *
        (np.sqrt(ch4) - np.sqrt(ch4_pi))
    )  * (1 + tropospheric_adjustment[:, :, :, [gas_index_mapping["CH4"]], :])

    # N2O
    erf_out[:, :, :, [gas_index_mapping["N2O"]], :] = (
        (a2*np.sqrt(co2) + b2*np.sqrt(n2o) + c2*np.sqrt(ch4) + d2) *
        (np.sqrt(n2o) - np.sqrt(n2o_pi))
    )  * (1 + tropospheric_adjustment[:, :, :, [gas_index_mapping["N2O"]], :])

    # Then, linear forcing for other gases
    minor_gas_index = list(range(concentration.shape[SPECIES_AXIS]))
    for major_gas in ['CO2', 'CH4', 'N2O']:
        minor_gas_index.remove(gas_index_mapping[major_gas])
    if len(minor_gas_index) > 0:
        erf_out[:, :, :, minor_gas_index, :] = (
            (concentration[:, :, :, minor_gas_index, :] - pre_industrial_concentration[:, :, :, minor_gas_index, :])
            * radiative_efficiency[:, :, :, minor_gas_index, :] * 0.001
        ) * (1 + tropospheric_adjustment[:, :, :, minor_gas_index, :])

    return erf_out


def calculate_erfari_forcing(
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
        self.species = map_species_scenario_config(self.scenarios, self.configs, self.run_config)
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
        self.aci_index = []
        #self.config_indices = []
        for ispec, specie in enumerate(self.species):
            self.species_index_mapping[specie.name] = ispec
            if specie.category in GREENHOUSE_GAS:
                self.ghg_indices.append(ispec)
            if specie.category in AEROSOL:
                self.ari_indices.append(ispec)
            if specie.category == Category.AEROSOL_CLOUD_INTERACTIONS:
                self.aci_index = ispec
        for iscen, scenario in enumerate(self.scenarios):
            self.scenarios_index_mapping[scenario.name] = iscen
        for iconf, config in enumerate(self.configs):
            self.configs_index_mapping[config.name] = iconf

    def _fill_concentration(self):
        """After the emissions to concentrations step we want to put the concs into each GreenhouseGas"""
        for ispec, species_name in enumerate(self.species_index_mapping):
            if self.species[ispec].category in GREENHOUSE_GAS: # and self.species[ispec].run_mode == RunMode.EMISSIONS: # don't think necessary as should just be replacing with same thing. We want the alpha for CH4 though.
                for iscen, scenario_name in enumerate(self.scenarios_index_mapping):
                    scen_spec = self.scenarios[iscen].list_of_species[ispec]
                    scen_spec.concentration = self.concentration_array[:, iscen, :, ispec, 0]
                    scen_spec.cumulative_emissions = np.squeeze(self.cumulative_emissions_array[:, iscen, :, ispec, 0])
                    scen_spec.airborne_fraction = self.airborne_emissions_array[:, iscen, :, ispec, 0] / self.cumulative_emissions_array[:, iscen, :, ispec, 0]
                    if self.species[ispec].category == Category.CH4:
                        scen_spec.effective_lifetime = self.alpha_lifetime_array[:, iscen, :, ispec, 0] * self.lifetime_array[:, 0, :, ispec, 0]


    def _fill_forcing(self):
        """Add the forcing as an attribute to each Species and Scenario"""
        for iscen, scenario in enumerate(self.scenarios):
            for ispec, specie in enumerate(self.species):
                self.scenarios[scenario].species[specie].forcing = self.forcing_array[:, iscen, :, ispec, 0]
            self.scenarios[scenario].forcing = self.forcing_sum_array[:, iscen, :, 0, 0]
            self.scenarios[scenario].stochastic_forcing = self.stochastic_forcing[:, iscen, :]


    def _fill_temperature(self):
        """Add the temperature as an attribute to each Scenario"""
        for iscen, scenario in enumerate(self.scenarios):
            self.scenarios[scenario].temperature_layers = self.temperature[:, iscen, :, 0, :]
            self.scenarios[scenario].temperature = self.temperature[:, iscen, :, 0, 0]

    def _initialise_arrays(self, n_timesteps, n_scenarios, n_configs, n_species):
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
        self.lifetime_array = np.ones((1, 1, n_configs, n_species, self.run_config.n_gas_boxes)) * np.nan
        self.baseline_emissions_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.baseline_concentration_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.partition_fraction_array = np.zeros((1, 1, n_configs, n_species, self.run_config.n_gas_boxes))
        self.natural_emissions_adjustment_array = np.zeros((1, 1, n_configs, n_species, 1))
        self.radiative_efficiency_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.forcing_scaling_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.efficacy_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.erfari_emissions_to_forcing_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        #self.scale_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        #self.shape_sulfur_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        #self.shape_bcoc_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.temperature = np.ones((n_timesteps, n_scenarios, n_configs, 1, self.run_config.n_temperature_boxes)) * np.nan

        for ispec, species_name in enumerate(self.species_index_mapping):
            for iconf, config_name in enumerate(self.configs_index_mapping):
                conf_spec = self.configs[iconf].species_configs[ispec]
                self.forcing_scaling_array[:, 0, iconf, ispec, 0] = (1+conf_spec.tropospheric_adjustment) * conf_spec.scale
                self.efficacy_array[:, 0, iconf, ispec, 0] = conf_spec.efficacy
                self.baseline_emissions_array[:, :, iconf, ispec, :] = conf_spec.baseline_emissions
                if self.species[ispec].category in GREENHOUSE_GAS:
                    self.lifetime_array[:, :, iconf, ispec, :] = conf_spec.lifetime
                    self.partition_fraction_array[:, :, iconf, ispec, :] = conf_spec.partition_fraction
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
                if self.species[ispec].category in HALOGEN:  # TODO: probably needs similar to above here.
                    self.fractional_release_array[:, :, iconf, ispec, 0] = conf_spec.fractional_release
                    self.br_atoms_array[:, :, iconf, ispec, 0] = conf_spec.br_atoms
                    self.cl_atoms_array[:, :, iconf, ispec, 0] = conf_spec.cl_atoms
                if self.species[ispec].category in AEROSOL:
                    self.erfari_emissions_to_forcing_array[:, 0, iconf, ispec, :] = conf_spec.erfari_emissions_to_forcing
            for iscen, scenario_name in enumerate(self.scenarios_index_mapping):
                scen_spec = self.scenarios[iscen].list_of_species[ispec]
                if self.species[ispec].run_mode == RunMode.EMISSIONS:
                    self.emissions_array[:, iscen, :, ispec, 0] = scen_spec.emissions[:, None]
                if self.species[ispec].run_mode == RunMode.CONCENTRATION:
                    self.concentration_array[:, iscen, :, ispec, 0] = scen_spec.concentration[:, None]
                if self.species[ispec].run_mode == RunMode.FORCING:
                    self.forcing_array[:, iscen, :, ispec, 0] = scen_spec.forcing[:, None]

                    # if isinstance(self.species[specie], AerosolCloudInteractions):
                    #     if hasattr(self.configs[iconfig], 'species') and specie in self.configs[iconfig].species:
                    #         self.scale_array[:, 0, iconfig, ispec, :] = self.configs[iconfig].species[specie].scale
                    #         self.shape_sulfur_array[:, 0, iconfig, ispec, :] = self.configs[iconfig].species[specie].shape_sulfur
                    #         self.shape_bcoc_array[:, 0, iconfig, ispec, :] = self.configs[iconfig].species[specie].shape_bcoc
                    #     else:
                    #         self.scale_array[:, 0, iconfig, ispec, :] = self.scenarios[scenario].species[specie].scale
                    #         self.shape_sulfur_array[:, 0, iconfig, ispec, :] = self.scenarios[scenario].species[specie].shape_sulfur
                    #         self.shape_bcoc_array[:, 0, iconfig, ispec, :] = self.scenarios[scenario].species[specie].shape_bcoc



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
        self._initialise_arrays(n_timesteps, n_scenarios, n_configs, n_species)
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

            if 'Aerosol-Cloud Interactions' in self.species_index_mapping:
                self.forcing_array[i_timestep:i_timestep+1, :, :, self.aci_index, :] = erf_aci(
                    self.emissions_array[[i_timestep], ...],
                    self.pre_industrial_emissions_array,
                    self.tropospheric_adjustment_array,
                    self.scale_array,
                    self.shape_sulfur_array,
                    self.shape_bcoc_array,
                    self.species_index_mapping
                )[0:1, :, :, self.aci_index, :]

            # ozone here
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

            # 100. run the energy balance model
            # TODO: skip if temperature prescribed
            # TODO: remove loop
            for iscen, scenario in enumerate(self.scenarios):
                for iconf, config in enumerate(self.configs):
                    temperature_boxes[0, iscen, iconf, 0, :] = (
                        eb_matrix_d[iconf] @ temperature_boxes[0, iscen, iconf, 0, :] +
                        forcing_vector_d[iconf] * self.forcing_sum_array[i_timestep, iscen, iconf, 0, 0] +
                        stochastic_d[iconf][i_timestep, :]
                    )
                    self.temperature[i_timestep, iscen, iconf, :, :] = temperature_boxes[0, iscen, iconf, 0, 1:]
                    self.stochastic_forcing[i_timestep, iscen, iconf] = temperature_boxes[0, iscen, iconf, 0, 0]

        self._fill_concentration()
#        self._fill_forcing()
#        self._fill_temperature()








### main
#for i in range(751):
