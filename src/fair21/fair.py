from collections.abc import Iterable
import copy
from dataclasses import dataclass, field
import typing

import numpy as np


# each Scenario will contain a list of Species
# each Config will contain settings, as well as options relating to each Species
# FAIR can contain one or more Scenarios and one or more Configs

IIRF_HORIZON = 100


@dataclass
class Species():
    name: str


@dataclass
class IIRF():
    iirf_0: float=field(default=None)
    iirf_airborne: float=0
    iirf_cumulative: float=0
    iirf_temperature: float=0


@dataclass
class Emissions():
    species: Species
    emissions: Iterable=None
    baseline: float=0
    natural_emissions_adjustment: float=0

    def __post_init__(self):
        if not isinstance(self.species, Species):
            raise ValueError(f"{self.species} is not of type Species")
        self.emissions = np.asarray(self.emissions)


@dataclass
class Concentration():
    species: Species
    concentration: typing.Union[tuple, list, np.ndarray]
    baseline: float=0


@dataclass
class Scenario():
    emissions: typing.List[Emissions]


co2 = Species(name="CO2")
ch4 = Species("CH4")
so2 = Species("Sulfur")

co2_e1 = Emissions(co2, np.arange(50))
ch4_e1 = Emissions(ch4, np.ones(50)*200)
co2_e2 = Emissions(co2, np.arange(50)*2)
ch4_e2 = Emissions(ch4, np.ones(50)*100)
co2_e3 = Emissions(co2, np.ones(50)*10)
ch4_e3 = Emissions(ch4, np.zeros(50))
so2_e  = Emissions(so2, np.ones(50)*100)

scen1 = Scenario((co2_e1, ch4_e1, so2_e))
scen2 = Scenario((co2_e2, ch4_e2, so2_e))
scen3 = Scenario((co2_e3, ch4_e3, so2_e))


@dataclass
class ClimateResponse():
    ocean_heat_capacity: typing.Union[Iterable, float]
    ocean_heat_transfer: typing.Union[Iterable, float]
    deep_ocean_efficacy: float=1
    stochastic_run: bool=False
    sigma_xi: float=None
    sigma_eta: float=None
    gamma_autocorrelation: float=None
    seed: int=None

    # type checking will take too long
    def __post_init__(self):
        self.ocean_heat_capacity=np.asarray(self.ocean_heat_capacity)
        self.ocean_heat_transfer=np.asarray(self.ocean_heat_transfer)


@dataclass
class GasConfig():
    species: Species
    molecular_weight: float
    lifetime: typing.Union[float, tuple, list, np.ndarray]
    partition_fraction: typing.Union[float, tuple, list, np.ndarray]
    radiative_efficiency: float=None
    iirf: IIRF=None
    g0: float=field(init=False)
    g1: float=field(init=False)
    scale: float=1
    tropospheric_adjustment: float=0
    efficacy: float=1

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

        if self.iirf is None:
            iirf_0 = (
                np.sum(np.asarray(self.lifetime) *
                (1 - np.exp(-IIRF_HORIZON / np.asarray(self.lifetime)))
                * np.asarray(self.partition_fraction))
            )
            self.iirf=IIRF(iirf_0, 0, 0, 0)


@dataclass
class AerosolConfig():
    species: Species
    erfari_emissions_to_forcing: float
    scale: float=1
    tropospheric_adjustment: float=0
    efficacy: float=1


@dataclass
class Config():
    climate_response: ClimateResponse
    species_config: typing.List[typing.Union[GasConfig, AerosolConfig]]

    def __post_init__(self):
        # check eveything provided is a Config
        if not hasattr(self.species_config, "__iter__"):
            self.species_config = [self.species_config]



cr1 = ClimateResponse(ocean_heat_capacity = (5, 20, 100), ocean_heat_transfer=(1,2,1), deep_ocean_efficacy=1.29)
cr2 = ClimateResponse(ocean_heat_capacity = (2, 10, 80), ocean_heat_transfer=(1,2,3))

# from default_configs import CO2, CH4, ...

#@dataclass
#class CO2(GasConfig):
#    species: Species=Species("CO2")
#    ... all the rest


co2_cfg1 = GasConfig(
    co2,
    lifetime=[1e9, 394, 36, 4.3],
    partition_fraction=[0.25, 0.25, 0.25, 0.25],
    radiative_efficiency=1.33e-5,
    molecular_weight=44.009,
    scale=1.04
)

co2_cfg2 = GasConfig(
    co2,
    lifetime=[1e9, 394, 36, 4.3],
    partition_fraction=[0.25, 0.25, 0.25, 0.25],
    radiative_efficiency=1.33e-5,
    molecular_weight=44.009,
    iirf=IIRF(29, 0.000819, 0.00846, 4),
    scale=0.99
)
co2_cfg1.radiative_efficiency=1.6e-5

ch4_cfg1 = GasConfig(
    ch4,
    lifetime=8.25,
    partition_fraction=1,
    molecular_weight=16.043,
    iirf=IIRF(8.25, 0.00032, 0, -0.3)
)
ch4_cfg2 = copy.copy(ch4_cfg1)
ch4_cfg2.scale=0.86


so2_cfg = AerosolConfig(so2, erfari_emissions_to_forcing=-0.00362)

config1 = Config(cr1, (co2_cfg1, ch4_cfg1, so2_cfg))
config2 = Config(cr2, (co2_cfg2, ch4_cfg2, so2_cfg))

print(config1)
print()
print(config2)

print()



# def _check_scenarios(scenarios):
#     scenarios = list(self.scenarios.keys())
# species_included = self.scenarios[scenarios[0]].species.keys()
# for scenario in scenarios[1:]:
#     if self.scenarios[scenario].species.keys() != species_included:
#         raise ValueError('put a better exception here')


class FAIR():
    def __init__(
        self,
        scenarios: typing.List[Scenario],
        configs: typing.List[Config],
        time: np.ndarray=None
    ):
        self.scenarios = scenarios
        self.configs = configs
        if time is not None:
            self.define_time(time)

        #_check_scenarios(self.scenarios)


    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(Scenario)


    def add_config(self, config: Config):
        self.configs.append(Config)


    def define_time(self, time: np.ndarray):
        if not hasattr(time, '__iter__'):
            raise TimeNotIterableError("Time must be an iterable data type")
        self.time = np.asarray(time)


    def _pre_run_checks(self):
        # Check if time vector is defined
        if not hasattr(self, 'time'):
            raise TimeNotDefinedError("Time vector is not defined")
        # Check if at least one scenario is defined
        if not hasattr(self, 'scenarios'):
            raise ScenariosNotDefinedError("Scenarios are not defined")
        # if more than one scenario is defined, check they have the same species
        if len(self.scenarios) > 1
        # # ensure all the scenarios contain the same species
        # if hasattr(self, 'scenarios'):
        #     if hasattr(self.scenarios, '__iter__') and len(self.scenarios) > 1:



    def run(self):
        # run initial sense checks on inputs.
        self._pre_run_checks()

        self.species = self.scenarios[list(self.scenarios.keys())[0]].species
        n_species = len(self.species)
        n_configs = len(self.configs)
        n_scenarios = len(self.scenarios)
        n_timesteps = len(self.time)
        self.time_deltas = _make_time_deltas(self.time)

        # initialise arrays. Using arrays makes things run quicker
        self._assign_indices()
        self._initialise_arrays(n_timesteps, n_scenarios, n_configs, n_species)
        gas_boxes = np.zeros((1, n_scenarios, n_configs, n_species, n_gas_boxes))
        temperature_boxes = np.zeros((1, n_scenarios, n_configs, 1, n_temperature_boxes+1))
        self.temperature_prescribed=False
#        if self.temperature_prescribed:
#            temperature_boxes = self.temperature[0, :]

        # initialise the energy balance model and get critical vectors
        # which itself needs to be run once per "config" and dimensioned correctly
        ebm_matrix_d = {}
        forcing_vector_d = {}
        stochastic_d = {}
        for config in self.configs:
            ebm = EnergyBalanceModel(
                ocean_heat_capacity=config.ocean_heat_capacity,
                ocean_heat_transfer=config.ocean_heat_transfer,
                deep_ocean_efficacy=config.deep_ocean_efficacy,
                stochastic_run=config.stochastic_run,
                sigma_eta=config.sigma_eta,
                sigma_xi=config.sigma_xi,
                gamma_autocorrelation=config.gamma_autocorrelation,
                seed=config.seed,
                n_timesteps=n_timesteps,
            )
            ebm_matrix_d[config] = ebm.eb_matrix_d
            forcing_vector_d[config] = ebm.forcing_vector_d
            stochastic_d[config] = ebm.stochastic_d

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
            alpha_lifetime_array[np.isnan(alpha_lifetime_array)]=1  # CF4 seems to have an issue. Should we raise warning?
            self.concentration_array[[i_timestep], ...], gas_boxes, self.airborne_emissions_array[[i_timestep], ...] = step_concentration(
                self.emissions_array[[i_timestep], ...],
                gas_boxes,
                self.airborne_emissions_array[[i_timestep-1], ...],
                self.burden_per_emission_array,
                self.lifetime_array,
                alpha_lifetime=alpha_lifetime_array,
                pre_industrial_concentration=self.pre_industrial_concentration_array,
                timestep=self.time_deltas[i_timestep],
                partition_fraction=self.partition_fraction_array,
                natural_emissions_adjustment=self.natural_emissions_adjustment_array,
            )
            self.alpha_lifetime_array[[i_timestep], ...] = alpha_lifetime_array

            # 2. concentrations to emissions for ghg emissions:
            # TODO:

            # 3. Greenhouse gas concentrations to forcing
            self.forcing_array[i_timestep:i_timestep+1, :, :, self.ghg_indices] = ghg(
                self.concentration_array[[i_timestep], ...],
                self.pre_industrial_concentration_array,
                self.tropospheric_adjustment_array,
                self.radiative_efficiency_array,
                self.species_index_mapping
            )[0:1, :, :, self.ghg_indices, :]

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


fair=FAIR(scenarios=(scen1, scen2, scen3), configs=(config1, config2))
print(fair)









@dataclass
class Forcing():
    species: Species
    forcing: typing.Union[tuple, list, np.ndarray]
    tropospheric_adjustment: float=0
    scale: float=1






### main
#for i in range(751):
