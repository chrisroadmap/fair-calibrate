import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from .constants import TIME_AXIS, SPECIES_AXIS, GASBOX_AXIS
from .earth_params import earth_radius, mass_atmosphere, seconds_per_year
from .energy_balance_model import step_temperature, calculate_toa_imbalance_postrun, multi_ebm
from .forcing.aerosol.erfari import calculate_erfari_forcing
from .forcing.aerosol.erfaci import stevens2015 as calculate_erfaci_forcing
from .forcing.ghg import meinshausen2020 as calculate_ghg_forcing
from .gas_cycle import calculate_alpha
from .gas_cycle.forward import step_concentration
from .gas_cycle.inverse import unstep_concentration
from .structure.species import species_types, valid_input_modes, multiple_allowed

# TODO:
# the smith2021 and meinshausen2020 imports can be put inside if statements
# lower in the code, which will allow for alternative treatments.

class FAIR:
    def __init__(self):
        pass


    def define_time(self, start, end, step):
        self.timebounds = np.arange(start, end+step, step)
        self.timepoints = 0.5 * (self.timebounds[1:] + self.timebounds[:-1])
        self.timestep = step
        self._n_timebounds = len(self.timebounds)
        self._n_timepoints = len(self.timepoints)


    def define_scenarios(self, scenarios):
        self.scenarios = scenarios
        self._n_scenarios = len(scenarios)


    def define_configs(self, configs):
        self.configs = configs
        self._n_configs = len(configs)


    def define_species(self, species, properties):
        self.species = species
        self._n_species = len(species)
        self.properties = properties


        # 1. everything we want to run with defined?
        for specie in species:
            if specie not in properties:
                raise ValueError(f"{specie} does not have a corresponding key in `properties`.")

            # 2. everything a valid species type?
            if properties[specie]['type'] not in species_types:
                raise ValueError(f"{properties[specie]['type']} is not a valid species type. Valid types are: {[t for t in species_types]}")

            # 3. input_modes valid?
            if properties[specie]['input_mode'] not in valid_input_modes[properties[specie]['type']]:
                raise ValueError(f"{properties[specie]['input_mode']} is not a valid input mode for {properties[specie]['type']}. Valid input modes are: {[m for m in valid_input_modes[properties[specie]['type']]]}")


    def run_control(
        self,
        n_gasboxes=4,
        n_layers=3,
        iirf_max=100,
        aci_method='smith2021',
        ghg_method='meinshausen2020',
        ch4_method='leach2021'
    ):
        self.gasboxes = range(n_gasboxes)
        self.layers = range(n_layers)
        self.iirf_max = iirf_max
        aci_method=aci_method.lower()
        if aci_method in ['smith2021', 'stevens2015']:
            self.aci_method = aci_method
        else:
            raise ValueError(f"aci_method should be smith2021 or stevens2015; you provided {aci_method}.")
        if ghg_method in ['leach2021', 'meinshausen2020', 'etminan2016', 'myhre1998']:
            self.ghg_method = ghg_method
        else:
            raise ValueError(f"`ghg_method` should be one of [leach2021, meinshausen2020, etminan2016, myhre1998]; you provided {ghg_method}.")
        self._n_gasboxes = n_gasboxes
        self._n_layers = n_layers
        #self._n_aci_parameters = 3 if aci_method=='smith2021' else 2
        self.aci_parameters = ['scale', 'Sulfur', 'BC+OC']#[:self._n_aci_parameters]


    def allocate(self):
        # driver/output variables
        self.emissions = xr.DataArray(
            np.ones((self._n_timepoints, self._n_scenarios, self._n_configs, self._n_species)) * np.nan,
            coords = (self.timepoints, self.scenarios, self.configs, self.species),
            dims = ('timepoints', 'scenario', 'config', 'specie')
        )
        self.concentration = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_species)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs, self.species),
            dims = ('timebounds', 'scenario', 'config', 'specie')
        )
        self.forcing = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_species)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs, self.species),
            dims = ('timebounds', 'scenario', 'config', 'specie')
        )
        self.temperature = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_layers)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs, self.layers),
            dims = ('timebounds', 'scenario', 'config', 'layer')
        )

        # output variables
        self.airborne_emissions = xr.DataArray(
            np.zeros((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_species)),
            coords = (self.timebounds, self.scenarios, self.configs, self.species),
            dims = ('timebounds', 'scenario', 'config', 'specie')
        )
        self.alpha_lifetime = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_species)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs, self.species),
            dims = ('timebounds', 'scenario', 'config', 'specie')
        )
        self.cumulative_emissions = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_species)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs, self.species),
            dims = ('timebounds', 'scenario', 'config', 'specie')
        )
        self.airborne_fraction = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_species)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs, self.species),
            dims = ('timebounds', 'scenario', 'config', 'specie')
        )
        self.ocean_heat_content_change = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs),
            dims = ('timebounds', 'scenario', 'config')
        )
        self.toa_imbalance = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs),
            dims = ('timebounds', 'scenario', 'config')
        )
        self.ocean_heat_content_change = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs),
            dims = ('timebounds', 'scenario', 'config')
        )
        self.stochastic_forcing = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs),
            dims = ('timebounds', 'scenario', 'config')
        )
        self.forcing_sum = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords = (self.timebounds, self.scenarios, self.configs),
            dims = ('timebounds', 'scenario', 'config')
        )

        # climate configs
        self.climate_configs = xr.Dataset(
            {
                'ocean_heat_transfer': (["config", "layer"], np.ones((self._n_configs, self._n_layers)) * np.nan),
                'ocean_heat_capacity': (["config", "layer"], np.ones((self._n_configs, self._n_layers)) * np.nan),
                'deep_ocean_efficacy': ("config", np.ones(self._n_configs) * np.nan),
                'stochastic_run': ("config", np.zeros(self._n_configs, dtype=bool)),
                'sigma_eta': ("config", np.ones(self._n_configs) * 0.5),
                'sigma_xi': ("config", np.ones(self._n_configs) * 0.5),
                'gamma_autocorrelation': ("config", np.ones(self._n_configs) * 2),
                'seed': ("config", np.zeros(self._n_configs, dtype=np.uint32)),
                'use_seed': ("config", np.zeros(self._n_configs, dtype=bool)),
                'forcing_4co2': ("config", np.ones(self._n_configs) * 8),
            },
            coords = {
                "config": self.configs,
                "layer": self.layers
            },
        )

        # species configs
        self.species_configs = xr.Dataset(
            {
                # general parameters applicable to all species
                'tropospheric_adjustment': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'forcing_efficacy': (["config", "specie"], np.ones((self._n_configs, self._n_species))),
                'forcing_temperature_feedback': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'forcing_scale': (["config", "specie"], np.ones((self._n_configs, self._n_species))),

                # greenhouse gas parameters
                'partition_fraction': (
                    ["config", "specie", "gasbox"], np.ones((self._n_configs, self._n_species, self._n_gasboxes)) * np.nan
                ),
                'unperturbed_lifetime': (
                    ["config", "specie", "gasbox"], np.ones((self._n_configs, self._n_species, self._n_gasboxes)) * np.nan
                ),
                'molecular_weight': ("specie", np.ones(self._n_species) * np.nan),
                'baseline_concentration': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),
                'iirf_0': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),
                'iirf_airborne': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),
                'iirf_uptake': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),
                'iirf_temperature': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),
                'baseline_emissions': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'g0': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),
                'g1': (["config", "specie"], np.ones((self._n_configs, self._n_species)) * np.nan),

                # general parameters relating emissions, concentration or forcing of one species to forcing of another
                # these are all linear factors
                'greenhouse_gas_radiative_efficiency': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'contrails_radiative_efficiency': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'erfari_radiative_efficiency': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'h2o_stratospheric_factor': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'lapsi_radiative_efficiency': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'land_use_cumulative_emissions_to_forcing': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),
                'ozone_radiative_efficiency': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),

                # specific parameters for ozone-depleting GHGs
                'cl_atoms': ("specie", np.zeros(self._n_species)),
                'br_atoms': ("specie", np.zeros(self._n_species)),
                'fractional_release': (["config", "specie"], np.zeros((self._n_configs, self._n_species))),

                # specific parameters for aerosol-cloud interactions
                'aci_parameters': (["config", "aci_parameter"], np.ones((self._n_configs, 3)) * np.nan)

            },
            coords = {
                "config": self.configs,
                "specie": self.species,
                "gasbox": self.gasboxes,
                "aci_parameter": self.aci_parameters
            },
        )


    # greenhouse gas convenience functions
    def calculate_iirf0(self, iirf_horizon=100):
        gasbox_axis = self.species_configs["partition_fraction"].get_axis_num('gasbox')
        self.species_configs["iirf_0"] = (
            np.sum(self.species_configs["unperturbed_lifetime"] *
            (1 - np.exp(-iirf_horizon / self.species_configs["unperturbed_lifetime"]))
            * self.species_configs["partition_fraction"], gasbox_axis)
        )


    def calculate_g(self, iirf_horizon=100):
        gasbox_axis = self.species_configs["partition_fraction"].get_axis_num('gasbox')
        self.species_configs["g1"] = np.sum(
            self.species_configs["partition_fraction"] * self.species_configs["unperturbed_lifetime"] *
            (1 - (1 + iirf_horizon/self.species_configs["unperturbed_lifetime"]) *
            np.exp(-iirf_horizon/self.species_configs["unperturbed_lifetime"])),
        axis=gasbox_axis)
        self.species_configs["g0"] = np.exp(-1 * np.sum((self.species_configs["partition_fraction"])*
            self.species_configs["unperturbed_lifetime"]*
            (1 - np.exp(-iirf_horizon/self.species_configs["unperturbed_lifetime"])), axis=gasbox_axis)/
            self.species_configs["g1"]
        )


    def calculate_concentration_per_emission(self, mass_atmosphere=5.1352e18, molecular_weight_air=28.97):
        self.species_configs["concentration_per_emission"] = 1 / (
            mass_atmosphere / 1e18 *
            self.species_configs["molecular_weight"] / molecular_weight_air
        )


    # climate response
    def _make_ebms(self):
        self.ebms = multi_ebm(
            self.configs,
            ocean_heat_capacity=self.climate_configs['ocean_heat_capacity'],
            ocean_heat_transfer=self.climate_configs['ocean_heat_transfer'],
            deep_ocean_efficacy=self.climate_configs['deep_ocean_efficacy'],
            stochastic_run=self.climate_configs['stochastic_run'],
            sigma_eta=self.climate_configs['sigma_eta'],
            sigma_xi=self.climate_configs['sigma_xi'],
            gamma_autocorrelation=self.climate_configs['gamma_autocorrelation'],
            seed=self.climate_configs['seed'],
            use_seed=self.climate_configs['use_seed'],
            forcing_4co2=self.climate_configs['forcing_4co2'],
            timestep=self.timestep,
            timebounds=self.timebounds,
        )


    def _check_properties(self):
        def _raise_if_nan(specie, input_mode):
            raise ValueError(f"{specie} contains NaN values in its {input_mode} array, which you are trying to drive the simulation with.")

        # check if emissions, concentration, forcing have been defined
        for specie in self.species:
            # 4. do we have non-nan data in every case?
            if self.properties[specie]['input_mode'] == 'emissions':
                n_nan = np.isnan(self.emissions.loc[dict(specie=specie)]).sum()
                if n_nan > 0: _raise_if_nan(specie, 'emissions')
            elif self.properties[specie]['input_mode'] == 'concentration':
                n_nan = np.isnan(self.concentration.loc[dict(specie=specie)]).sum()
                if n_nan > 0: _raise_if_nan(specie, 'concentration')
            elif self.properties[specie]['input_mode'] == 'forcing':
                n_nan = np.isnan(self.forcing.loc[dict(specie=specie)]).sum()
                if n_nan > 0: _raise_if_nan(specie, 'forcing')

        properties_df = pd.DataFrame(self.properties).T.reindex(self.species)

        # 5. special dependency cases
        if 'co2' in list(properties_df.loc[properties_df['input_mode']=='calculated']['type']):
            if (
                'co2 ffi' not in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']) or
                'co2 afolu' not in list(properties_df.loc[properties_df['input_mode']=='emissions']['type'])
            ):
                raise ValueError('`co2` in `calculated` mode requires `co2 ffi` and `co2 afolu` in `emissions` mode.')

        if 'land use' in list(properties_df.loc[properties_df['input_mode']=='calculated']['type']):
            if 'co2 afolu' not in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']):
                raise ValueError('`land use` in `calculated` mode requires `co2 afolu` in `emissions` mode.')

        if 'aerosol-cloud interactions' in list(properties_df.loc[properties_df['input_mode']=='calculated']['type']):
            if 'sulfur' not in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']):
                raise ValueError('`aerosol-cloud interactions` in `calculated` mode requires `sulfur` in `emissions` mode for `aci_method = stevens2015`.')
            elif (
                self.aci_method=='smith2021' and
                'black carbon' not in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']) and
                'organic carbon' not in list(properties_df.loc[properties_df['input_mode']=='emissions']['type'])
            ):
                raise ValueError('`aerosol-cloud interactions` in `calculated` mode requires `sulfur`, `black carbon` and `organic carbon` in `emissions` mode for `aci_method = smith2021`.')

        co2_to_forcing = False
        ch4_to_forcing = False
        n2o_to_forcing = False

        if 'co2' in list(properties_df.loc[properties_df['input_mode']=='calculated']['type']) or 'co2' in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']) or 'co2' in list(properties_df.loc[properties_df['input_mode']=='concentration']['type']):
            co2_to_forcing=True
        if 'ch4' in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']) or 'ch4' in list(properties_df.loc[properties_df['input_mode']=='concentration']['type']):
            ch4_to_forcing=True
        if 'n2o' in list(properties_df.loc[properties_df['input_mode']=='emissions']['type']) or 'n2o' in list(properties_df.loc[properties_df['input_mode']=='concentration']['type']):
            n2o_to_forcing=True
        if self.ghg_method in ['meinshausen2020', 'etminan2016']:
            if 0 < co2_to_forcing+ch4_to_forcing+n2o_to_forcing < 3:
                raise ValueError("For `ghg_method` either `meinshausen2016` or `etminan2016`, either all of `co2`, `ch4` and `n2o` must be provided in a form that can be converted to concentrations, or none")
        elif self.ghg_method=='myhre1998':
            if 0 < ch4_to_forcing+n2o_to_forcing < 2:
                raise ValueError("For `ghg_method` either `myhre1998`, either both of `ch4` and `n2o` must be provided in a form that can be converted to concentrations, or neither")

        # 6. uniques
        for specie_type in properties_df['type'].unique():
            n_repeats = sum(properties_df['type']==specie_type)
            if n_repeats > 1 and not multiple_allowed[specie_type]:
                raise ValueError(f'{specie_type} is defined {n_repeats} times in the problem, but must be unique.')

        self.properties_df = properties_df


    def _make_indices(self):
        # the following are all n_species-length boolean arrays

        # these define what we utimately want input or output from
        self._emissions_species = list(self.properties_df.loc[self.properties_df.loc[:,'emissions']==True].index)
        self._concentration_species = list(self.properties_df.loc[self.properties_df.loc[:,'concentration']==True].index)
        self._forcing_species = list(self.properties_df.loc[self.properties_df.loc[:,'forcing']==True].index)

        # these define which species do what in FaIR
        self._ghg_indices = np.asarray(self.properties_df.loc[:, 'greenhouse_gas'].values, dtype=bool)
        self._ari_precursor_indices = np.asarray(self.properties_df.loc[:, 'aerosol_radiation_precursor'].values, dtype=bool)
        self._aci_precursor_indices = np.asarray(self.properties_df.loc[:, 'aerosol_cloud_precursor'].values, dtype=bool)
        self._co2_ffi_indices = np.asarray(self.properties_df['type']=='co2 ffi', dtype=bool)
        self._co2_afolu_indices = np.asarray(self.properties_df['type']=='co2 afolu', dtype=bool)
        self._co2_indices = np.asarray(self.properties_df['type']=='co2', dtype=bool)
        self._ch4_indices = np.asarray(self.properties_df['type']=='ch4', dtype=bool)
        self._n2o_indices = np.asarray(self.properties_df['type']=='n2o', dtype=bool)
        self._sulfur_indices = np.asarray(self.properties_df['type']=='sulfur', dtype=bool)
        self._bc_indices = np.asarray(self.properties_df['type']=='black carbon', dtype=bool)
        self._oc_indices = np.asarray(self.properties_df['type']=='organic carbon', dtype=bool)
        self._ari_indices = np.asarray(self.properties_df['type']=='aerosol-radiation interactions', dtype=bool)
        self._aci_indices = np.asarray(self.properties_df['type']=='aerosol-cloud interactions', dtype=bool)
        self._minor_ghg_indices = self._ghg_indices ^ self._co2_indices ^ self._ch4_indices ^ self._n2o_indices

        # and these ones are more specific, tripping certain behaviours or functions
        self._ghg_forward_indices = np.asarray(
            (
                (
                    (self.properties_df.loc[:,'input_mode']=='emissions')|
                    (self.properties_df.loc[:,'input_mode']=='calculated')
                ) &
                (self.properties_df.loc[:,'greenhouse_gas'])
            ).values, dtype=bool
        )
        self._ghg_inverse_indices = np.asarray(
            (
                (self.properties_df.loc[:,'input_mode']=='concentration')&
                (self.properties_df.loc[:,'greenhouse_gas'])
            ).values, dtype=bool
        )
        self._ari_from_emissions_indices = np.asarray(
            (
                ~(self.properties_df.loc[:,'greenhouse_gas'])&
                (self.properties_df.loc[:,'aerosol_radiation_precursor'])
            ).values, dtype=bool
        )
        self._ari_from_concentration_indices = np.asarray(
            (
                (self.properties_df.loc[:,'greenhouse_gas'])&
                (self.properties_df.loc[:,'aerosol_radiation_precursor'])
            ).values, dtype=bool
        )


    def run(self, progress=True):
        self._check_properties()
        self._make_indices()
        self._make_ebms()

        # part of pre-run: TODO move to a new method
        if self._co2_indices.sum() + self._co2_ffi_indices.sum() + self._co2_afolu_indices.sum()==3:
            self.emissions[...,self._co2_indices] = self.emissions[...,self._co2_ffi_indices].data + self.emissions[...,self._co2_afolu_indices].data
        self.cumulative_emissions[1:,...] = self.emissions.cumsum(axis=0, skipna=False) * self.timestep + self.cumulative_emissions[0,...]

        # create numpy arrays
        alpha_lifetime_array = self.alpha_lifetime.data
        airborne_emissions_array = self.airborne_emissions.data
        baseline_concentration_array = self.species_configs['baseline_concentration'].data
        baseline_emissions_array = self.species_configs['baseline_emissions'].data
        concentration_array = self.concentration.data
        concentration_per_emission_array = self.species_configs['concentration_per_emission'].data
        cummins_state_array = np.ones((self._n_timebounds, self._n_scenarios, self._n_configs, self._n_layers+1)) * np.nan
        cumulative_emissions_array = self.cumulative_emissions.data
        deep_ocean_efficacy_array = self.climate_configs['deep_ocean_efficacy'].data
        eb_matrix_d_array = self.ebms['eb_matrix_d'].data
        emissions_array = self.emissions.data
        erfari_radiative_efficiency_array = self.species_configs['erfari_radiative_efficiency'].data
        erfaci_scale_array = self.species_configs['aci_parameters'].data[:,0]
        erfaci_shape_sulfur_array = self.species_configs['aci_parameters'].data[:,1]
        erfaci_shape_bcoc_array = self.species_configs['aci_parameters'].data[:,2]
        forcing_array = self.forcing.data
        forcing_scale_array = self.species_configs['forcing_scale'].data
        forcing_efficacy_array = self.species_configs['forcing_efficacy'].data
        forcing_efficacy_sum_array = np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan
        forcing_sum_array = self.forcing_sum.data
        forcing_vector_d_array = self.ebms['forcing_vector_d'].data
        g0_array = self.species_configs['g0'].data
        g1_array = self.species_configs['g1'].data
        gas_partitions_array = np.zeros((self._n_scenarios, self._n_configs, self._n_species, self._n_gasboxes))
        greenhouse_gas_radiative_efficiency_array = self.species_configs['greenhouse_gas_radiative_efficiency'].data
        iirf_0_array = self.species_configs['iirf_0'].data
        iirf_airborne_array = self.species_configs['iirf_airborne'].data
        iirf_temperature_array = self.species_configs['iirf_temperature'].data
        iirf_uptake_array = self.species_configs['iirf_uptake'].data
        iirf_temperature = self.species_configs['iirf_temperature'].data
        ocean_heat_transfer_array = self.climate_configs['ocean_heat_transfer'].data
        partition_fraction_array = self.species_configs['partition_fraction'].data
        stochastic_d_array = self.ebms['stochastic_d'].data
        unperturbed_lifetime_array = self.species_configs['unperturbed_lifetime'].data

        # forcing should be initialised so this should not be nan. We could check, or allow silent fail as some species don't take forcings and would correctly be nan.
        forcing_sum_array[0:1, ...] = np.nansum(
            forcing_array[0:1, ...], axis=SPECIES_AXIS
        )

        # this is the most important state vector
        cummins_state_array[0, ..., 0] = forcing_sum_array[0, ...]
        cummins_state_array[..., 1:] = self.temperature.data

        # it's all been leading to this : FaIR MAIN LOOP
        for i_timepoint in tqdm(range(self._n_timepoints), disable=1-progress, desc="Timestepping"):

            # 1. alpha scaling
            alpha_lifetime_array[i_timepoint:i_timepoint+1, ..., self._ghg_indices] = calculate_alpha(   # this timepoint
                airborne_emissions_array[i_timepoint:i_timepoint+1, ..., self._ghg_indices],  # last timebound
                cumulative_emissions_array[i_timepoint:i_timepoint+1, ..., self._ghg_indices],  # last timebound
                g0_array[None, None, ..., self._ghg_indices],
                g1_array[None, None, ..., self._ghg_indices],
                iirf_0_array[None, None, ..., self._ghg_indices],
                iirf_airborne_array[None, None, ..., self._ghg_indices],
                iirf_temperature_array[None, None, ..., self._ghg_indices],
                iirf_uptake_array[None, None, ..., self._ghg_indices],
                cummins_state_array[i_timepoint:i_timepoint+1, ..., 1:2],
                self.iirf_max
            )

            # 2. methane lifetime here
            # 3. emissions to concentrations
            (
                concentration_array[i_timepoint+1:i_timepoint+2, ..., self._ghg_forward_indices],
                gas_partitions_array[..., self._ghg_forward_indices, :],
                airborne_emissions_array[i_timepoint+1:i_timepoint+2, ..., self._ghg_forward_indices]
            ) = step_concentration(
                emissions_array[i_timepoint:i_timepoint+1, ..., self._ghg_forward_indices, None],  # this timepoint
                gas_partitions_array[..., self._ghg_forward_indices, :], # last timebound
                airborne_emissions_array[i_timepoint+1:i_timepoint+2, ..., self._ghg_forward_indices, None],  # last timebound
                alpha_lifetime_array[i_timepoint:i_timepoint+1, ..., self._ghg_forward_indices, None],
                baseline_concentration_array[None, None, ..., self._ghg_forward_indices],
                baseline_emissions_array[None, None, ..., self._ghg_forward_indices, None],
                concentration_per_emission_array[None, None, ..., self._ghg_forward_indices],
                unperturbed_lifetime_array[None, None, ..., self._ghg_forward_indices, :],
        #        oxidation_matrix,
                partition_fraction_array[None, None, ..., self._ghg_forward_indices, :],
                self.timestep,
            )
            # 4. concentrations to emissions
            (
                emissions_array[i_timepoint:i_timepoint+1, ..., self._ghg_inverse_indices],
                gas_partitions_array[..., self._ghg_inverse_indices, :],
                airborne_emissions_array[i_timepoint+1:i_timepoint+2, ..., self._ghg_inverse_indices]
            ) = unstep_concentration(
                concentration_array[i_timepoint+1:i_timepoint+2, ..., self._ghg_inverse_indices],  # this timepoint
                gas_partitions_array[None, ..., self._ghg_inverse_indices, :], # last timebound
                airborne_emissions_array[i_timepoint:i_timepoint+1, ..., self._ghg_inverse_indices, None],  # last timebound
                alpha_lifetime_array[i_timepoint:i_timepoint+1, ..., self._ghg_inverse_indices, None],
                baseline_concentration_array[None, None, ..., self._ghg_inverse_indices],
                baseline_emissions_array[None, None, ..., self._ghg_inverse_indices],
                concentration_per_emission_array[None, None, ..., self._ghg_inverse_indices],
                unperturbed_lifetime_array[None, None, ..., self._ghg_inverse_indices, :],
        #        oxidation_matrix,
                partition_fraction_array[None, None, ..., self._ghg_inverse_indices, :],
                self.timestep,
            )
            cumulative_emissions_array[i_timepoint+1, ..., self._ghg_inverse_indices] = cumulative_emissions_array[i_timepoint, ..., self._ghg_inverse_indices] + emissions_array[i_timepoint, ..., self._ghg_inverse_indices] * self.timestep

            # 5. greenhouse gas concentrations to forcing
            forcing_array[i_timepoint+1:i_timepoint+2, ..., self._ghg_indices] = calculate_ghg_forcing(
                concentration_array[i_timepoint+1:i_timepoint+2, ...],
                baseline_concentration_array[None, None, ...] * np.ones((1, self._n_scenarios, self._n_configs, self._n_species)),
                forcing_scale_array[None, None, ...],
                greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )[0:1, ..., self._ghg_indices]

            # 6. aerosol direct forcing
            forcing_array[i_timepoint+1:i_timepoint+2, ..., self._ari_indices] = calculate_erfari_forcing(
                emissions_array[i_timepoint:i_timepoint+1, ...],
                concentration_array[i_timepoint+1:i_timepoint+2, ...],
                baseline_emissions_array[None, None, ...],
                baseline_concentration_array[None, None, ...],
                forcing_scale_array[None, None, ...],
                erfari_radiative_efficiency_array[None, None, ...],
                self._ari_from_emissions_indices,
                self._ari_from_concentration_indices,
            )

            # 7. aerosol indirect forcing
            forcing_array[i_timepoint+1:i_timepoint+2, ..., self._aci_indices] = calculate_erfaci_forcing(
                emissions_array[i_timepoint:i_timepoint+1, ...],
                baseline_emissions_array[None, None, ...],
                forcing_scale_array[None, None, ..., self._aci_indices],
                erfaci_scale_array[None, None, :, None],
                erfaci_shape_sulfur_array[None, None, :, None],
                #erfaci_shape_bcoc_array[None, None, :, None],
                self._sulfur_indices,
                #self._bc_indices,
                #self._oc_indices,
                #self.aci_method
            )

            # 8. ozone
            # 9. contrails from NOx
            # 10. BC and OC to LAPSI
            # 11. CH4 to stratospheric water vapour
            # 12. CO2 cumulative to land use change
            # 13. volcanic forcing temperature dependence?
            # 14. sum forcings
            forcing_sum_array[i_timepoint+1:i_timepoint+2, ...] = np.nansum(
                forcing_array[i_timepoint+1:i_timepoint+2, ...], axis=SPECIES_AXIS
            )
            forcing_efficacy_sum_array[i_timepoint+1:i_timepoint+2, ...]=np.nansum(
                forcing_array[i_timepoint+1:i_timepoint+2, ...]*forcing_efficacy_array[None, None, ...], axis=SPECIES_AXIS
            )

            # 15. forcing to temperature
            #if not self.run_config.temperature_prescribed:
            cummins_state_array[i_timepoint+1:i_timepoint+2, ...] = step_temperature(
                cummins_state_array[i_timepoint:i_timepoint+1, ...],
                eb_matrix_d_array[None, None, ...],
                forcing_vector_d_array[None, None, ...],
                stochastic_d_array[i_timepoint+1:i_timepoint+2, None, ...],
                forcing_efficacy_sum_array[i_timepoint+1:i_timepoint+2, ..., None]
            )

        # 16. TOA imbalance
        toa_imbalance_array = calculate_toa_imbalance_postrun(
            cummins_state_array,
            forcing_sum_array,#[..., None],  # not efficacy adjusted, is this correct?
            ocean_heat_transfer_array,
            deep_ocean_efficacy_array,
        )

        # 17. Ocean heat content change
        ocean_heat_content_change_array = (
            np.cumsum(toa_imbalance_array * self.timestep, axis=TIME_AXIS) * earth_radius**2 * 4 * np.pi * seconds_per_year
        )

        # 18. calculate airborne fraction - we have NaNs and zeros we know about, and we don't mind
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            airborne_fraction_array = airborne_emissions_array / cumulative_emissions_array

        # 19. (Re)allocate to xarray
        self.temperature.data = cummins_state_array[..., 1:]
        self.concentration.data = concentration_array
        self.emissions.data = emissions_array
        self.forcing.data = forcing_array
        self.forcing_sum.data = forcing_sum_array
        self.cumulative_emissions.data = cumulative_emissions_array
        self.airborne_emissions.data = airborne_emissions_array
        self.airborne_fraction.data = airborne_fraction_array
        self.ocean_heat_content_change.data = ocean_heat_content_change_array
        self.toa_imbalance.data = toa_imbalance_array
        self.stochastic_forcing.data = cummins_state_array[..., 0]
