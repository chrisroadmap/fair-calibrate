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

class Category(Enum):
    """Types of Species encountered in climate scenarios."""
    CO2 = auto()
    CH4 = auto()
    N2O = auto()
    HALOGEN = auto()
    F_GAS = auto()
    AEROSOL = auto()
    OZONE_PRECURSOR = auto()
    OZONE = auto()
    AEROSOL_CLOUD_INTERACTIONS = auto()
    CONTRAILS = auto()
    BC_ON_SNOW = auto()
    LAND_USE = auto()
    VOLCANIC = auto()
    SOLAR = auto()

GREENHOUSE_GAS = [Category.CO2, Category.CH4, Category.N2O, Category.HALOGEN, Category.F_GAS]


@dataclass
class Species():
    name: str
    category: Category


@dataclass
class Emissions():
    species: Species
    emissions: np.ndarray=None
    baseline_emissions: float=0
    natural_emissions_adjustment: float=0


@dataclass
class Concentration():
    species: Species
    concentration: np.ndarray=None
    baseline_concentration: float=0


@dataclass
class Forcing():
    species: Species
    forcing: np.ndarray=None


@dataclass
class Placeholder():
    species: Species


@dataclass
class IIRF():
    iirf_0: float=field(default=None)
    iirf_airborne: float=0
    iirf_cumulative: float=0
    iirf_temperature: float=0


@dataclass
class RunConfig():
    n_gas_boxes = 4
    n_temperature_boxes = 3
    temperature_prescribed = False


@dataclass
class GasProperties():
    molecular_weight: float=None
    lifetime: typing.Union[float, tuple, list, np.ndarray]=None
    partition_fraction: typing.Union[float, tuple, list, np.ndarray]=None
    radiative_efficiency: float=None
    iirf: IIRF=None

    def __post_init__(self):
        # auto-fill based on name
        # validate input
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
        if self.iirf is None:
            iirf_0 = (
                np.sum(np.asarray(self.lifetime) *
                (1 - np.exp(-IIRF_HORIZON / np.asarray(self.lifetime)))
                * np.asarray(self.partition_fraction))
            )
            self.iirf=IIRF(iirf_0, 0, 0, 0)


@dataclass
class ForcingProperties():
    tropospheric_adjustment: float=0
    scale: float=1
    efficacy: float=1


@dataclass
class AerosolProperties():
    erfari_emissions_to_forcing: float=0
    lapsi_emissions_to_forcing: float=0
# define ERFaci as the Config() level
#    erfaci_beta: float=None
#    erfaci_shape_sulfur: float=None
#    erfaci_shape_bcoc: float=None

@dataclass
class OzoneProperties():
    ozone_radiative_efficiency: float=None
    cl_atoms: float=None
    br_atoms: float=None
    fractional_release: float=None


@dataclass
class SpeciesConfig():
    species: Species
    gas_properties: GasProperties=None
    aerosol_properties: AerosolProperties=None
    ozone_properties: OzoneProperties=None
    forcing_properties: ForcingProperties=ForcingProperties()

    def __post_init__(self):
        if self.species.category == Category.HALOGEN:
            if ~isinstance(self.ozone_properties, OzoneProperties):
                raise MissingInputError("For Halogens, ozone_properties needs to be supplied to SpeciesConfig and be of type OzoneProperties")
            if ~isinstance(self.ozone_properties.ozone_radiative_efficiency, Number):
                raise ValueError("ozone_properties.ozone_radiative_efficiency should be a number for Halogens")
            if ~isinstance(self.ozone_properties.cl_atoms, int) or self.cl_atoms < 0:
                raise ValueError("ozone_properties.cl_atoms should be a non-negative integer for Halogens")
            if ~isinstance(self.ozone_properties.br_atoms, int) or self.cl_atoms < 0:
                raise ValueError("ozone_properties.br_atoms should be a non-negative integer for Halogens")
            if ~isinstance(self.ozone_properties.fractional_release, Number) or self.ozone_properties.fractional_release < 0:
                raise ValueError("ozone_properties.fractional_release should be a non-negative number for Halogens")


@dataclass
class Scenario():
    name: str
    list_of_species: typing.List[Species]

    def __post_init__(self):
        if not isinstance(self.list_of_species, list):
            raise TypeError('list_of_species argument passed to Scenario must be a list of Species')

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

    def __post_init__(self):
        # check eveything provided is a Config
        if not isinstance(self.species_configs, list):
            raise TypeError('species_configs argument passed to Config must be a list of SpeciesConfig')


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
                        if hasattr(scenarios[0].list_of_species[ispec], attr):
                            n_timesteps_first_scenario_species = len(getattr(scenarios[0].list_of_species[ispec], attr))
                species_included_first_scenario.append(scenarios[0].list_of_species[ispec].species)
                for attr in ('emissions', 'concentration', 'forcing'):
                    if hasattr(scenarios[0].list_of_species[ispec], attr):
                        n_timesteps_this_scenario_species = len(getattr(scenarios[0].list_of_species[ispec], attr))
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
            if scenarios[iscen].list_of_species[ispec].species in species_included:
                raise DuplicationError(
                    f"{scenarios[iscen].list_of_species[ispec].species.name} "
                    f"is duplicated in a Scenario."
                )
            species_included.append(scenarios[iscen].list_of_species[ispec].species)
            for attr in ('emissions', 'concentration', 'forcing'):
                if hasattr(scenarios[iscen].list_of_species[ispec], attr):
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
                species_included_first_config.append(configs[0].species_configs[ispec].species)
        species_included = []
        n_species = len(configs[iconf].species_configs)
        for ispec in range(n_species):
            species_included.append(configs[iconf].species_configs[ispec].species)
        if species_included != species_included_first_config:
            raise SpeciesMismatchError(
                f"Each Config must contain the same list of SpeciesConfigs, in the "
                f"same order")


def map_species_scenario_config(scenarios, configs):
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
    """
    # at this point we have checked self-consistency, so we only need to check
    # the first element of each.
    species_included_first_config = []
    n_species_first_config = len(configs[0].species_configs)
    for ispec, species in enumerate(configs[0].species_configs):
        species_included_first_config.append(configs[0].species_configs[ispec].species)
    species_included_first_scenario = []
    n_species_first_scenario = len(scenarios[0].list_of_species)
    for ispec, species in enumerate(scenarios[0].list_of_species):
        species_included_first_scenario.append(scenarios[0].list_of_species[ispec].species)
    if species_included_first_config != species_included_first_scenario:
        raise SpeciesMismatchError(
            f"The list of Species provided to Scenario.list_of_species is "
            f"{[species.name for species in species_included_first_scenario]}. "
            f"This differs from that provided to Config.species_configs "
            f"{[species.name for species in species_included_first_config]}."
        )

    # now that we have a unique list of species, check for Categories for which
    # it makes no sense to duplicate
    no_dupes = [
        Category.CO2,
        Category.CH4,
        Category.N2O,
        Category.OZONE,
        Category.AEROSOL_CLOUD_INTERACTIONS,
        Category.CONTRAILS,
        Category.BC_ON_SNOW,
        Category.LAND_USE,
        Category.SOLAR,
        Category.VOLCANIC
    ]

    running_total = {category: 0 for category in no_dupes}
    for species in species_included_first_config:
        if species.category in no_dupes:
            running_total[species.category] =+ 1
            if running_total[species.category] > 1:
                raise DuplicationError(
                    f"The scenario contains more than one instance of "
                    f"{species.category}"
                )
    n_major_ghgs = running_total[Category.CO2] + running_total[Category.CH4] + running_total[Category.N2O]
    if 0 < n_major_ghgs < 3: #TODO and emissions or concentration driven mode
        raise IncompatibleConfigError(
            f"Either all of CO2, CH4 and N2O must be given in a Scenario, or "
            f"none, unless running purely forcing-driven. If you want to "
            f"exclude the effect of one or two of these gases, consider setting "
            f"emissions of these gases to zero."
        )
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


def _make_ebm(configs):
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
#            n_timesteps=n_timesteps,
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


def ghg(
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
    # TODO: what happens if a scenario does not include all these gases?
    print(gas_index_mapping)
    # Check whether all of CO2, CH4 and N2O are provided. If they are not,
    # TODO: Raise a warning
    # and set to default baseline


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
        self.species = map_species_scenario_config(self.scenarios, self.configs)
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
            if specie.category == Category.AEROSOL:
                self.ari_indices.append(ispec)
            if specie.category == Category.AEROSOL_CLOUD_INTERACTIONS:
                self.aci_index = ispec
        for iscen, scenario in enumerate(self.scenarios):
            self.scenarios_index_mapping[scenario.name] = iscen
        for iconf, config in enumerate(self.configs):
            self.configs_index_mapping[config.name] = iconf
        print(self.species_index_mapping)
        print(self.scenarios_index_mapping)
        print(self.configs_index_mapping)

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
        self.baseline_emissions_array = np.ones((1, n_scenarios, n_configs, n_species, 1)) * np.nan  # revisit
        self.baseline_concentration_array = np.ones((1, n_scenarios, n_configs, n_species, 1)) * np.nan # revisit
        self.partition_fraction_array = np.zeros((1, 1, n_configs, n_species, self.run_config.n_gas_boxes))
        self.natural_emissions_adjustment_array = np.zeros((1, n_scenarios, n_configs, n_species, 1))
        self.radiative_efficiency_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.forcing_scaling_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.efficacy_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        self.erfari_emissions_to_forcing_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        #self.scale_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        #self.shape_sulfur_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan
        #self.shape_bcoc_array = np.ones((1, 1, n_configs, n_species, 1)) * np.nan

        # START HERE AND GO BACK TO NAMES
        for ispec, species_name in enumerate(self.species_index_mapping):
            for iconf, config_name in enumerate(self.configs_index_mapping):
                conf_spec_gas = self.configs[iconf].species_configs[ispec].gas_properties
                conf_spec_aero = self.configs[iconf].species_configs[ispec].aerosol_properties
                conf_spec_forc = self.configs[iconf].species_configs[ispec].forcing_properties
                self.forcing_scaling_array[:, 0, iconf, ispec, 0] = (1+conf_spec_forc.tropospheric_adjustment) * conf_spec_forc.scale
                self.efficacy_array[:, 0, iconf, ispec, 0] = conf_spec_forc.efficacy
                if self.species[ispec].category in GREENHOUSE_GAS:
                    self.lifetime_array[:, 0, iconf, ispec, :] = conf_spec_gas.lifetime
                    self.partition_fraction_array[:, 0, iconf, ispec, :] = conf_spec_gas.partition_fraction
                    self.iirf_0_array[:, 0, iconf, ispec, :] = conf_spec_gas.iirf.iirf_0
                    self.iirf_cumulative_array[:, 0, iconf, ispec, :] = conf_spec_gas.iirf.iirf_cumulative
                    self.iirf_temperature_array[:, 0, iconf, ispec, :] = conf_spec_gas.iirf.iirf_temperature
                    self.iirf_airborne_array[:, 0, iconf, ispec, :] = conf_spec_gas.iirf.iirf_airborne
                    self.burden_per_emission_array[:, 0, iconf, ispec, :] = conf_spec_gas.burden_per_emission
                    self.radiative_efficiency_array[:, 0, iconf, ispec, :] = conf_spec_gas.radiative_efficiency
                    self.g0_array[:, 0, iconf, ispec, :] = conf_spec_gas.g0
                    self.g1_array[:, 0, iconf, ispec, :] = conf_spec_gas.g1
                if self.species[ispec].category == Category.HALOGEN:  # TODO: probably needs similar to above here.
                    self.fractional_release_array[:, 0, iconf, ispec, 0] = scen_spec.fractional_release
                    self.br_atoms_array[:, 0, iconf, ispec, 0] = scen_spec.br_atoms
                    self.cl_atoms_array[:, 0, iconf, ispec, 0] = scen_spec.cl_atoms
                if self.species[ispec].category == Category.AEROSOL:
                    self.erfari_emissions_to_forcing_array[:, 0, iconf, ispec, :] = conf_spec_aero.erfari_emissions_to_forcing
                for iscen, scenario_name in enumerate(self.scenarios_index_mapping):
                    scen_spec = self.scenarios[iscen].list_of_species[ispec]
                    if hasattr(scen_spec, 'emissions'):
                        self.emissions_array[:, iscen, iconf, ispec, 0] = scen_spec.emissions
                        self.baseline_emissions_array[:, iscen, 0, ispec, 0] = scen_spec.baseline_emissions
                    if hasattr(scen_spec, 'concentration'):
                        self.concentration_array[:, iscen, iconf, ispec, 0] = scen_spec.concentration
                        self.baseline_concentration_array[:, iscen, 0, ispec, 0] = scen_spec.baseline_concentration
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
        #self.temperature_prescribed=False
#        if self.temperature_prescribed:
#            temperature_boxes = self.temperature[0, :]

        # initialise the energy balance model and get critical vectors
        # which itself needs to be run once per "config" and dimensioned correctly

        eb_matrix_d, forcing_vector_d, stochastic_d = _make_ebm(self.configs)

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

#### START HERE: diagnosing why concentration_array is nan
            #print(gas_boxes.shape)
            print(self.burden_per_emission_array)
            import sys; sys.exit()

            # 2. concentrations to emissions for ghg emissions:
            # TODO:

            # 3. Greenhouse gas concentrations to forcing
            self.forcing_array[i_timestep:i_timestep+1, :, :, self.ghg_indices] = ghg(
                self.concentration_array[[i_timestep], ...],
                self.baseline_concentration_array,
                self.forcing_scaling_array,
                self.radiative_efficiency_array,
                self.species_index_mapping
            )[0:1, :, :, self.ghg_indices, :]
            print(self.forcing_array[i_timestep:i_timestep+1, :, :, :])


            # 4. aerosol emissions to forcing
            self.forcing_array[i_timestep:i_timestep+1, :, :, self.ari_indices, :] = erf_ari(
                self.emissions_array[[i_timestep], ...],
                self.pre_industrial_emissions_array,
                self.tropospheric_adjustment_array,
                self.ari_radiative_efficiency_array,
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

            # 100. run the energy balance model - if temperature not prescribed - updating temperature boxes
            #TODO: remove loop
            for iscen, scenario in enumerate(self.scenarios):
                for iconf, config in enumerate(self.configs):
                    temperature_boxes[0, iscen, iconf, 0, :] = (
                        ebm_matrix_d[config] @ temperature_boxes[0, iscen, iconf, 0, :] +
                        forcing_vector_d[config] * self.forcing_sum_array[i_timestep, iscen, iconf, 0, 0] +
                        stochastic_d[config][i_timestep, :]
                    )
                    self.temperature[i_timestep, iscen, iconf, :, :] = temperature_boxes[0, iscen, iconf, 0, 1:]
                    self.stochastic_forcing[i_timestep, iscen, iconf] = temperature_boxes[0, iscen, iconf, 0, 0]

        self._fill_concentration()
        self._fill_forcing()
        self._fill_temperature()








### main
#for i in range(751):
