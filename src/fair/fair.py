"""Finite-amplitude Impulse Response (FaIR) simple climate model."""

import copy
import os
import warnings

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from .constants import SPECIES_AXIS, TIME_AXIS
from .earth_params import (
    earth_radius,
    mass_atmosphere,
    molecular_weight_air,
    seconds_per_year,
)
from .energy_balance_model import (
    calculate_toa_imbalance_postrun,
    multi_ebm,
    step_temperature,
)
from .forcing.aerosol.erfaci import leach2021aci, smith2021, stevens2015
from .forcing.aerosol.erfari import calculate_erfari_forcing
from .forcing.ghg import etminan2016, leach2021ghg, meinshausen2020, myhre1998
from .forcing.minor import calculate_linear_forcing
from .forcing.ozone import thornhill2021
from .gas_cycle import calculate_alpha
from .gas_cycle.ch4_lifetime import calculate_alpha_ch4
from .gas_cycle.eesc import calculate_eesc
from .gas_cycle.forward import step_concentration
from .gas_cycle.inverse import unstep_concentration
from .interface import fill
from .structure.species import multiple_allowed, species_types, valid_input_modes
from .structure.units import (
    compound_convert,
    desired_concentration_units,
    desired_emissions_units,
    mixing_ratio_convert,
    prefix_convert,
    time_convert,
)

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SPECIES_CONFIG_FILE = os.path.join(
    HERE, "defaults", "data", "ar6", "species_configs_properties.csv"
)


class FAIR:
    """FaIR simple climate model [1]_, [2]_, [3]_.

    References
    ----------
    .. [1] Leach, N. J., Jenkins, S., Nicholls, Z., Smith, C. J., Lynch, J.,
        Cain, M., Walsh, T., Wu, B., Tsutsui, J., and Allen, M. R. (2021).
        FaIRv2.0.0: a generalized impulse response model for climate uncertainty
        and future scenario exploration. Geoscientific Model Development, 14,
        3007–3036

    .. [2] Smith, C. J., Forster, P. M.,  Allen, M., Leach, N., Millar, R. J.,
        Passerello, G. A., and Regayre, L. A. (2018). FAIR v1.3: a simple
        emissions-based impulse response and carbon cycle model, Geosci. Model
        Dev., 11, 2273–2297

    .. [3] Millar, R.J., Nicholls, Z.R., Friedlingstein, P., Allen, M.R. (2017).
        A modified impulse-response representation of the global near-surface
        air temperature and atmospheric concentration response to carbon dioxide
        emissions. Atmospheric Chemistry and Physics, 17, 7213-7228.
    """

    def __init__(
        self,
        n_gasboxes=4,
        n_layers=3,
        iirf_max=100,
        br_cl_ods_potential=45,
        aci_method="smith2021",
        ghg_method="meinshausen2020",
        ch4_method="leach2021",
        temperature_prescribed=False,
    ):
        """Initialise FaIR.

        Parameters
        ----------
        n_gasboxes : int, default=4
            the number of atmospheric greenhouse gas boxes to run the model with
        n_layers : int, default=3
            the number of ocean layers in the energy balance or impulse
            response model to run with
        iirf_max : float, default=100
            limit for time-integral of greenhouse gas impulse response function.
        br_cl_ods_potential : float, default=45
            factor describing the ratio of efficiency that each bromine atom
            has as an ozone depleting substance relative to each chlorine atom.
        aci_method : str, default="smith2021"
            method to use for calculating forcing from aerosol-cloud
            interactions. Valid methods are smith2021, leach2021 and
            stevens2015.
        ghg_method : str, default="meinshausen2020"
            method to use for calculating greenhouse gas forcing from CO2, CH4
            and N2O. Valid methods are leach2021, meinshausen2020,
            etminan2016 and myhre1998.
        ch4_method : str, default="leach2021"
            method to use for calculating methane lifetime change. Valid
            methods are leach2021 and thornhill2021.
        temperature_prescribed : bool, default=False
            Run FaIR with temperatures prescribed.

        Raises
        ------
        ValueError :
            if aci_method, ghg_method or ch4_method given are not valid methods.
        """
        self._aci_method = aci_method
        self._ghg_method = ghg_method
        self._ch4_method = ch4_method
        self.gasboxes = range(n_gasboxes)
        self.layers = range(n_layers)
        self.iirf_max = iirf_max
        self.br_cl_ods_potential = br_cl_ods_potential
        self._n_gasboxes = n_gasboxes
        self._n_layers = n_layers
        self.aci_parameters = ["scale", "Sulfur", "BC+OC"]
        self.temperature_prescribed = temperature_prescribed

    # must be a less cumbsersome way to code this
    @property
    def aci_method(self):
        """Return aerosol-cloud interactions forcing method."""
        return self._aci_method.lower()

    @aci_method.setter
    def aci_method(self, value):
        if value.lower() in ["smith2021", "stevens2015", "leach2021"]:
            self._aci_method = value.lower()
        else:
            raise ValueError(
                f"aci_method should be one of [smith2021, stevens2015, leach2021]; you "
                f"provided {value.lower()}."
            )

    @property
    def ch4_method(self):
        """Return methane lifetime method."""
        return self._ch4_method.lower()

    @ch4_method.setter
    def ch4_method(self, value):
        if value.lower() in ["thornhill2021", "leach2021"]:
            self._ch4_method = value.lower()
        else:
            raise ValueError(
                f"ch4_method should be thornhill2021 or leach2021; you provided "
                f"{value.lower()}."
            )

    @property
    def ghg_method(self):
        """Return greenhouse gas forcing method."""
        return self._ghg_method.lower()

    @ghg_method.setter
    def ghg_method(self, value):
        if value.lower() in [
            "leach2021",
            "meinshausen2020",
            "etminan2016",
            "myhre1998",
        ]:
            self._ghg_method = value.lower()
        else:
            raise ValueError(
                f"ghg_method should be one of [leach2021, meinshausen2020, "
                f"etminan2016, myhre1998]; you provided {value.lower()}."
            )

    def define_time(self, start, end, step):
        """Define timebounds vector to run FaIR.

        Parameters
        ----------
        start : float
            first timebound of the model (year)
        end : float
            last timebound of the model (year)
        step : float
            timestep (year)
        """
        self.timebounds = np.arange(start, end + step / 2, step)
        self.timepoints = 0.5 * (self.timebounds[1:] + self.timebounds[:-1])
        self.timestep = step
        self._n_timebounds = len(self.timebounds)
        self._n_timepoints = len(self.timepoints)

    def define_scenarios(self, scenarios):
        """Define scenarios to analyse in FaIR.

        Parameters
        ----------
        scenarios : list
            scenario names to run
        """
        self.scenarios = scenarios
        self._n_scenarios = len(scenarios)

    def define_configs(self, configs):
        """Define configs to analyse in FaIR.

        Parameters
        ----------
        configs : list
            config names to run
        """
        self.configs = configs
        self._n_configs = len(configs)

    def define_species(self, species, properties):
        """Define species to run in FaIR.

        Parameters
        ----------
        species : list
            names of species to include in FaIR
        properties : dict
            mapping of each specie to particular run properties. This is a
            nested dict, which must contain the five required entries.

        Raises
        ------
        ValueError :
            if a specie in species does not have a matching key in properties.
        ValueError :
            if an invalid species type is specified.
        ValueError :
            if an invalid input_type (driving mode) is provided for a particular
            type.
        ValueError :
            if duplicate species types are provided for types that must be
            unique.
        """
        self.species = species
        self._n_species = len(species)

        # everything we want to run with defined?
        for specie in species:
            if specie not in properties:
                raise ValueError(
                    f"{specie} does not have a corresponding key in `properties`."
                )

            # everything a valid species type?
            if properties[specie]["type"] not in species_types:
                raise ValueError(
                    f"{properties[specie]['type']} is not a valid species type. Valid "
                    f"types are: {[t for t in species_types]}"
                )

            # input_modes valid?
            if (
                properties[specie]["input_mode"]
                not in valid_input_modes[properties[specie]["type"]]
            ):
                raise ValueError(
                    f"{properties[specie]['input_mode']} is not a valid input mode for "
                    f"{properties[specie]['type']}. Valid input modes are: "
                    f"{[m for m in valid_input_modes[properties[specie]['type']]]}"
                )

        # on the way in, we don't mind if properties is over-specified, but
        # by the time we call allocate(), species and properties must align, so
        # we trim self.properties to match species.
        self.properties = properties
        self.properties_df = pd.DataFrame(self.properties).T.reindex(self.species)

        # 4. check that unique species actually are
        for specie_type in self.properties_df["type"].unique():
            n_repeats = sum(self.properties_df["type"] == specie_type)
            if n_repeats > 1 and not multiple_allowed[specie_type]:
                raise ValueError(
                    f"{specie_type} is defined {n_repeats} times in the problem, but "
                    f"must be unique."
                )

    def allocate(self):
        """Create `xarray`s of data input and output."""
        # check dimensions declared
        required_attributes_and_uncalled_method = {
            "_n_timepoints": "define_time()",
            "_n_scenarios": "define_scenarios()",
            "_n_configs": "define_configs()",
            "_n_species": "define_species()",
        }
        for attr, method in required_attributes_and_uncalled_method.items():
            if not hasattr(self, attr):
                raise AttributeError(
                    f"'FAIR' object has no attribute '{attr}'. Did you forget to call "
                    f"'{method}'?"
                )

        # driver/output variables
        self.emissions = xr.DataArray(
            np.ones(
                (
                    self._n_timepoints,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            )
            * np.nan,
            coords=(self.timepoints, self.scenarios, self.configs, self.species),
            dims=("timepoints", "scenario", "config", "specie"),
        )
        self.concentration = xr.DataArray(
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            )
            * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs, self.species),
            dims=("timebounds", "scenario", "config", "specie"),
        )
        self.forcing = xr.DataArray(
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            )
            * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs, self.species),
            dims=("timebounds", "scenario", "config", "specie"),
        )
        self.temperature = xr.DataArray(
            np.ones(
                (self._n_timebounds, self._n_scenarios, self._n_configs, self._n_layers)
            )
            * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs, self.layers),
            dims=("timebounds", "scenario", "config", "layer"),
        )

        # output variables
        self.airborne_emissions = xr.DataArray(
            np.zeros(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            ),
            coords=(self.timebounds, self.scenarios, self.configs, self.species),
            dims=("timebounds", "scenario", "config", "specie"),
        )
        self.alpha_lifetime = xr.DataArray(
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            )
            * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs, self.species),
            dims=("timebounds", "scenario", "config", "specie"),
        )
        self.cumulative_emissions = xr.DataArray(
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            )
            * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs, self.species),
            dims=("timebounds", "scenario", "config", "specie"),
        )
        self.airborne_fraction = xr.DataArray(
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_species,
                )
            )
            * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs, self.species),
            dims=("timebounds", "scenario", "config", "specie"),
        )
        self.ocean_heat_content_change = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs),
            dims=("timebounds", "scenario", "config"),
        )
        self.toa_imbalance = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs),
            dims=("timebounds", "scenario", "config"),
        )
        self.stochastic_forcing = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs),
            dims=("timebounds", "scenario", "config"),
        )
        self.forcing_sum = xr.DataArray(
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan,
            coords=(self.timebounds, self.scenarios, self.configs),
            dims=("timebounds", "scenario", "config"),
        )

        # climate configs
        self.climate_configs = xr.Dataset(
            {
                "ocean_heat_transfer": (
                    ["config", "layer"],
                    np.ones((self._n_configs, self._n_layers)) * np.nan,
                ),
                "ocean_heat_capacity": (
                    ["config", "layer"],
                    np.ones((self._n_configs, self._n_layers)) * np.nan,
                ),
                "deep_ocean_efficacy": ("config", np.ones(self._n_configs) * np.nan),
                "stochastic_run": ("config", np.zeros(self._n_configs, dtype=bool)),
                "sigma_eta": ("config", np.ones(self._n_configs) * 0.5),
                "sigma_xi": ("config", np.ones(self._n_configs) * 0.5),
                "gamma_autocorrelation": ("config", np.ones(self._n_configs) * 2),
                "seed": ("config", np.zeros(self._n_configs, dtype=np.uint32)),
                "use_seed": ("config", np.zeros(self._n_configs, dtype=bool)),
                "forcing_4co2": ("config", np.ones(self._n_configs) * 8),
            },
            coords={"config": self.configs, "layer": self.layers},
        )

        # species configs
        self.species_configs = xr.Dataset(
            {
                # general parameters applicable to all species
                # NB: at present forcing_scale has NO EFFECT on species provided
                # as prescribed forcing.
                "tropospheric_adjustment": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "forcing_efficacy": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)),
                ),
                "forcing_temperature_feedback": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "forcing_scale": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)),
                ),
                # greenhouse gas parameters
                "partition_fraction": (
                    ["config", "specie", "gasbox"],
                    np.ones((self._n_configs, self._n_species, self._n_gasboxes))
                    * np.nan,
                ),
                "unperturbed_lifetime": (
                    ["config", "specie", "gasbox"],
                    np.ones((self._n_configs, self._n_species, self._n_gasboxes))
                    * np.nan,
                ),
                "molecular_weight": ("specie", np.ones(self._n_species) * np.nan),
                "baseline_concentration": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "iirf_0": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "iirf_airborne": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "iirf_uptake": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "iirf_temperature": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "baseline_emissions": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "g0": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "g1": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "concentration_per_emission": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "forcing_reference_concentration": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                # general parameters relating emissions, concentration or forcing of one
                # species to forcing of another.
                # these are all linear factors
                "greenhouse_gas_radiative_efficiency": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "contrails_radiative_efficiency": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "erfari_radiative_efficiency": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "h2o_stratospheric_factor": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "lapsi_radiative_efficiency": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "land_use_cumulative_emissions_to_forcing": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                "ozone_radiative_efficiency": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                # specific parameters for ozone-depleting GHGs
                "cl_atoms": ("specie", np.zeros(self._n_species)),
                "br_atoms": ("specie", np.zeros(self._n_species)),
                "fractional_release": (
                    ["config", "specie"],
                    np.zeros((self._n_configs, self._n_species)),
                ),
                # specific parameters for methane lifetime
                "ch4_lifetime_chemical_sensitivity": (
                    ["config", "specie"],
                    np.ones((self._n_configs, self._n_species)) * np.nan,
                ),
                "lifetime_temperature_sensitivity": (
                    ["config"],
                    np.ones((self._n_configs)) * np.nan,
                ),
                # specific parameters for aerosol-cloud interactions
                "aci_parameters": (
                    ["config", "aci_parameter"],
                    np.ones((self._n_configs, 3)) * np.nan,
                ),
            },
            coords={
                "config": self.configs,
                "specie": self.species,
                "gasbox": self.gasboxes,
                "aci_parameter": self.aci_parameters,
            },
        )

    def fill_species_configs(self, filename=DEFAULT_SPECIES_CONFIG_FILE):
        """Fill the species_configs with values from a CSV file.

        Parameters
        ----------
        filename : str, optional
            Path of the CSV file to read the species configs from. If omitted, the
            default configs file will be read.
        """
        df = pd.read_csv(filename, index_col=0)
        for specie in self.species:
            fill(
                self.species_configs["tropospheric_adjustment"],
                df.loc[specie].tropospheric_adjustment,
                specie=specie,
            )
            fill(
                self.species_configs["forcing_efficacy"],
                df.loc[specie].forcing_efficacy,
                specie=specie,
            )
            fill(
                self.species_configs["forcing_temperature_feedback"],
                df.loc[specie].forcing_temperature_feedback,
                specie=specie,
            )
            fill(
                self.species_configs["forcing_scale"],
                df.loc[specie].forcing_scale,
                specie=specie,
            )
            for gasbox in range(self._n_gasboxes):
                fill(
                    self.species_configs["partition_fraction"],
                    df.loc[specie, f"partition_fraction{gasbox}"],
                    specie=specie,
                    gasbox=gasbox,
                )
                fill(
                    self.species_configs["unperturbed_lifetime"],
                    df.loc[specie, f"unperturbed_lifetime{gasbox}"],
                    specie=specie,
                    gasbox=gasbox,
                )
            fill(
                self.species_configs["molecular_weight"],
                df.loc[specie].molecular_weight,
                specie=specie,
            )
            fill(
                self.species_configs["baseline_concentration"],
                df.loc[specie].baseline_concentration,
                specie=specie,
            )
            fill(
                self.species_configs["forcing_scale"],
                df.loc[specie].forcing_scale,
                specie=specie,
            )
            fill(
                self.species_configs["forcing_reference_concentration"],
                df.loc[specie].forcing_reference_concentration,
                specie=specie,
            )
            fill(self.species_configs["iirf_0"], df.loc[specie].iirf_0, specie=specie)
            fill(
                self.species_configs["iirf_airborne"],
                df.loc[specie].iirf_airborne,
                specie=specie,
            )
            fill(
                self.species_configs["iirf_uptake"],
                df.loc[specie].iirf_uptake,
                specie=specie,
            )
            fill(
                self.species_configs["iirf_temperature"],
                df.loc[specie].iirf_temperature,
                specie=specie,
            )
            fill(
                self.species_configs["baseline_emissions"],
                df.loc[specie].baseline_emissions,
                specie=specie,
            )
            fill(self.species_configs["g0"], df.loc[specie].g0, specie=specie)
            fill(self.species_configs["g1"], df.loc[specie].g1, specie=specie)
            fill(
                self.species_configs["greenhouse_gas_radiative_efficiency"],
                df.loc[specie].greenhouse_gas_radiative_efficiency,
                specie=specie,
            )
            fill(
                self.species_configs["contrails_radiative_efficiency"],
                df.loc[specie].contrails_radiative_efficiency,
                specie=specie,
            )
            fill(
                self.species_configs["erfari_radiative_efficiency"],
                df.loc[specie].erfari_radiative_efficiency,
                specie=specie,
            )
            fill(
                self.species_configs["h2o_stratospheric_factor"],
                df.loc[specie].h2o_stratospheric_factor,
                specie=specie,
            )
            fill(
                self.species_configs["lapsi_radiative_efficiency"],
                df.loc[specie].lapsi_radiative_efficiency,
                specie=specie,
            )
            fill(
                self.species_configs["land_use_cumulative_emissions_to_forcing"],
                df.loc[specie].land_use_cumulative_emissions_to_forcing,
                specie=specie,
            )
            fill(
                self.species_configs["ozone_radiative_efficiency"],
                df.loc[specie].ozone_radiative_efficiency,
                specie=specie,
            )
            fill(
                self.species_configs["cl_atoms"], df.loc[specie].cl_atoms, specie=specie
            )
            fill(
                self.species_configs["br_atoms"], df.loc[specie].br_atoms, specie=specie
            )
            fill(
                self.species_configs["fractional_release"],
                df.loc[specie].fractional_release,
                specie=specie,
            )
            fill(
                self.species_configs["ch4_lifetime_chemical_sensitivity"],
                df.loc[specie].ch4_lifetime_chemical_sensitivity,
                specie=specie,
            )
        if "aci" in list(self.properties_df["type"]):
            fill(
                self.species_configs["aci_parameters"],
                df.loc["Aerosol-cloud interactions"].aci_params_scale,
                aci_parameter="scale",
            )
            fill(
                self.species_configs["aci_parameters"],
                df.loc["Aerosol-cloud interactions"].aci_params_Sulfur,
                aci_parameter="Sulfur",
            )
            fill(
                self.species_configs["aci_parameters"],
                df.loc["Aerosol-cloud interactions"].aci_params_BCOC,
                aci_parameter="BC+OC",
            )
        fill(
            self.species_configs["lifetime_temperature_sensitivity"],
            df.loc[df["type"] == "ch4"].lifetime_temperature_sensitivity,
        )
        self.calculate_concentration_per_emission()

    # greenhouse gas convenience functions
    def calculate_iirf0(self, iirf_horizon=100):
        r"""Convert greenhouse gas lifetime to time-integrated airborne fraction.

        iirf_0 is the 100-year time-integrated airborne fraction to a pulse
        emission. We know that the gas's atmospheric airborne fraction $a(t)$ for a
        gas with lifetime $\tau$ after time $t$ is therefore

        $a(t) = \exp(-t/tau)$

        and integrating this for 100 years after a pulse emissions gives us:

        $r_0(t) = \int_0^{100} \exp(-t/\tau) dt = \tau (1 - \exp (-100 / \tau))$.

        100 years is the default time horizon in FaIR but this can be set to any
        value.

        Parameters
        ----------
        iirf_horizon : float, optional, default=100
            time horizon for time-integrated airborne fraction (yr).
        """
        gasbox_axis = self.species_configs["partition_fraction"].get_axis_num("gasbox")
        self.species_configs["iirf_0"] = np.sum(
            self.species_configs["unperturbed_lifetime"]
            * (1 - np.exp(-iirf_horizon / self.species_configs["unperturbed_lifetime"]))
            * self.species_configs["partition_fraction"],
            gasbox_axis,
        )

    def calculate_g(self, iirf_horizon=100):
        """Calculate lifetime scaling parameters."""
        gasbox_axis = self.species_configs["partition_fraction"].get_axis_num("gasbox")
        self.species_configs["g1"] = np.sum(
            self.species_configs["partition_fraction"]
            * self.species_configs["unperturbed_lifetime"]
            * (
                1
                - (1 + iirf_horizon / self.species_configs["unperturbed_lifetime"])
                * np.exp(-iirf_horizon / self.species_configs["unperturbed_lifetime"])
            ),
            axis=gasbox_axis,
        )
        self.species_configs["g0"] = np.exp(
            -1
            * np.sum(
                (self.species_configs["partition_fraction"])
                * self.species_configs["unperturbed_lifetime"]
                * (
                    1
                    - np.exp(
                        -iirf_horizon / self.species_configs["unperturbed_lifetime"]
                    )
                ),
                axis=gasbox_axis,
            )
            / self.species_configs["g1"]
        )

    def calculate_concentration_per_emission(
        self, mass_atmosphere=mass_atmosphere, molecular_weight_air=molecular_weight_air
    ):
        """Calculate change in atmospheric concentration for each unit emission."""
        self.species_configs["concentration_per_emission"] = 1 / (
            mass_atmosphere
            / 1e18
            * self.species_configs["molecular_weight"]
            / molecular_weight_air
        )

    def fill_from_rcmip(self):
        """Fill emissions, concentrations and/or forcing from RCMIP scenarios."""
        # lookup converting FaIR default names to RCMIP names
        species_to_rcmip = {specie: specie.replace("-", "") for specie in self.species}
        species_to_rcmip["CO2 FFI"] = "CO2|MAGICC Fossil and Industrial"
        species_to_rcmip["CO2 AFOLU"] = "CO2|MAGICC AFOLU"
        species_to_rcmip["NOx aviation"] = "NOx|MAGICC Fossil and Industrial|Aircraft"
        species_to_rcmip[
            "Aerosol-radiation interactions"
        ] = "Aerosols-radiation interactions"
        species_to_rcmip[
            "Aerosol-cloud interactions"
        ] = "Aerosols-radiation interactions"
        species_to_rcmip["Contrails"] = "Contrails and Contrail-induced Cirrus"
        species_to_rcmip["Light absorbing particles on snow and ice"] = "BC on Snow"
        species_to_rcmip[
            "Stratospheric water vapour"
        ] = "CH4 Oxidation Stratospheric H2O"
        species_to_rcmip["Land use"] = "Albedo Change"

        species_to_rcmip_copy = copy.deepcopy(species_to_rcmip)

        for specie in species_to_rcmip_copy:
            if specie not in self.species:
                del species_to_rcmip[specie]

        rcmip_emissions_file = pooch.retrieve(
            url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
            known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
        )

        rcmip_concentration_file = pooch.retrieve(
            url="doi:10.5281/zenodo.4589756/rcmip-concentrations-annual-means-v5-1-0.csv",
            known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
        )

        rcmip_forcing_file = pooch.retrieve(
            url="doi:10.5281/zenodo.4589756/rcmip-radiative-forcing-annual-means-v5-1-0.csv",
            known_hash="md5:87ef6cd4e12ae0b331f516ea7f82ccba",
        )

        df_emis = pd.read_csv(rcmip_emissions_file)
        df_conc = pd.read_csv(rcmip_concentration_file)
        df_forc = pd.read_csv(rcmip_forcing_file)

        for scenario in self.scenarios:
            for specie, specie_rcmip_name in species_to_rcmip.items():
                if self.properties_df.loc[specie, "input_mode"] == "emissions":
                    # Grab raw emissions from dataframe
                    emis_in = (
                        df_emis.loc[
                            (df_emis["Scenario"] == scenario)
                            & (
                                df_emis["Variable"].str.endswith(
                                    "|" + specie_rcmip_name
                                )
                            )
                            & (df_emis["Region"] == "World"),
                            "1750":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                    # throw error if data missing
                    if emis_in.shape[0] == 0:
                        raise ValueError(
                            f"I can't find a value for scenario={scenario}, variable "
                            f"name ending with {specie_rcmip_name} in the RCMIP "
                            f"emissions database."
                        )

                    # RCMIP are "annual averages"; for emissions this is basically
                    # the emissions over the year, for concentrations and forcing
                    # it would be midyear values. In every case, we can assume
                    # midyear values and interpolate to our time grid.
                    rcmip_index = np.arange(1750.5, 2501.5)
                    interpolator = interp1d(
                        rcmip_index,
                        emis_in,
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    emis = interpolator(self.timepoints)

                    # We won't throw an error if the time is out of range for RCMIP,
                    # but we will fill with NaN to allow a user to manually specify
                    # pre- and post- emissions.
                    emis[self.timepoints < 1750] = np.nan
                    emis[self.timepoints > 2501] = np.nan

                    # Parse and possibly convert unit in input file to what FaIR wants
                    unit = df_emis.loc[
                        (df_emis["Scenario"] == scenario)
                        & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_emis["Region"] == "World"),
                        "Unit",
                    ].values[0]
                    emis = emis * (
                        prefix_convert[unit.split()[0]][
                            desired_emissions_units[specie].split()[0]
                        ]
                        * compound_convert[unit.split()[1].split("/")[0]][
                            desired_emissions_units[specie].split()[1].split("/")[0]
                        ]
                        * time_convert[unit.split()[1].split("/")[1]][
                            desired_emissions_units[specie].split()[1].split("/")[1]
                        ]
                    )  # * self.timestep

                    # fill FaIR xarray
                    fill(
                        self.emissions, emis[:, None], specie=specie, scenario=scenario
                    )

                if self.properties_df.loc[specie, "input_mode"] == "concentration":
                    # Grab raw concentration from dataframe
                    conc_in = (
                        df_conc.loc[
                            (df_conc["Scenario"] == scenario)
                            & (
                                df_conc["Variable"].str.endswith(
                                    "|" + specie_rcmip_name
                                )
                            )
                            & (df_conc["Region"] == "World"),
                            "1700":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                    # throw error if data missing
                    if conc_in.shape[0] == 0:
                        raise ValueError(
                            f"I can't find a value for scenario={scenario}, variable "
                            f"name ending with {specie_rcmip_name} in the RCMIP "
                            f"concentration database."
                        )

                    # interpolate: this time to timebounds
                    rcmip_index = np.arange(1700.5, 2501.5)
                    interpolator = interp1d(
                        rcmip_index,
                        conc_in,
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    conc = interpolator(self.timebounds)

                    # strip out pre- and post-
                    conc[self.timebounds < 1700] = np.nan
                    conc[self.timebounds > 2501] = np.nan

                    # Parse and possibly convert unit in input file to what FaIR wants
                    unit = df_conc.loc[
                        (df_conc["Scenario"] == scenario)
                        & (df_conc["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_conc["Region"] == "World"),
                        "Unit",
                    ].values[0]
                    conc = conc * (
                        mixing_ratio_convert[unit][desired_concentration_units[specie]]
                    )

                    # fill FaIR xarray
                    fill(
                        self.concentration,
                        conc[:, None],
                        specie=specie,
                        scenario=scenario,
                    )

                if self.properties_df.loc[specie, "input_mode"] == "forcing":
                    # Grab raw concentration from dataframe
                    forc_in = (
                        df_forc.loc[
                            (df_forc["Scenario"] == scenario)
                            & (
                                df_forc["Variable"].str.endswith(
                                    "|" + specie_rcmip_name
                                )
                            )
                            & (df_forc["Region"] == "World"),
                            "1750":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                    # throw error if data missing
                    if forc_in.shape[0] == 0:
                        raise ValueError(
                            f"I can't find a value for scenario={scenario}, variable "
                            f"name ending with {specie_rcmip_name} in the RCMIP "
                            f"radiative forcing database."
                        )

                    # interpolate: this time to timebounds
                    rcmip_index = np.arange(1750.5, 2501.5)
                    interpolator = interp1d(
                        rcmip_index,
                        forc_in,
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    forc = interpolator(self.timebounds)

                    # strip out pre- and post-
                    forc[self.timebounds < 1750] = np.nan
                    forc[self.timebounds > 2501] = np.nan

                    # Forcing so far is always W m-2, but perhaps this will change.

                    # fill FaIR xarray
                    fill(self.forcing, forc[:, None], specie=specie, scenario=scenario)

    # climate response
    def _make_ebms(self):
        # First check for NaNs
        for var in [
            "ocean_heat_capacity",
            "ocean_heat_transfer",
            "deep_ocean_efficacy",
            "gamma_autocorrelation",
        ]:
            if np.isnan(self.climate_configs[var]).sum() > 0:
                raise ValueError(
                    f"There are NaN values in FAIR.climate_configs['{var}']"
                )
        if self.climate_configs["stochastic_run"].sum() > 0:
            for var in ["sigma_eta", "sigma_xi", "seed"]:
                if np.isnan(self.climate_configs[var]).sum() > 0:
                    raise ValueError(
                        f"There are NaN values in climate_configs['{var}'], which is "
                        f"not allowed for FAIR.climate_configs['stochastic_run']=True"
                    )

        self.ebms = multi_ebm(
            self.configs,
            ocean_heat_capacity=self.climate_configs["ocean_heat_capacity"],
            ocean_heat_transfer=self.climate_configs["ocean_heat_transfer"],
            deep_ocean_efficacy=self.climate_configs["deep_ocean_efficacy"],
            stochastic_run=self.climate_configs["stochastic_run"],
            sigma_eta=self.climate_configs["sigma_eta"],
            sigma_xi=self.climate_configs["sigma_xi"],
            gamma_autocorrelation=self.climate_configs["gamma_autocorrelation"],
            seed=self.climate_configs["seed"],
            use_seed=self.climate_configs["use_seed"],
            forcing_4co2=self.climate_configs["forcing_4co2"],
            timestep=self.timestep,
            timebounds=self.timebounds,
        )

    def _check_properties(self):
        def _raise_if_nan(specie, input_mode):
            raise ValueError(
                f"{specie} contains NaN values in its {input_mode} array, which you "
                f"are trying to drive the simulation with."
            )

        self._routine_flags = {
            "ghg": False,
            "ari": False,
            "aci": False,
            "eesc": False,
            "contrails": False,
            "ozone": False,
            "land use": False,
            "lapsi": False,
            "h2o stratospheric": False,
            "temperature": True,
        }
        # check if emissions, concentration, forcing have been defined and
        # that we have non-nan data in every case
        for specie in self.species:
            if self.properties[specie]["input_mode"] == "emissions":
                n_nan = np.isnan(self.emissions.loc[dict(specie=specie)]).sum()
                if n_nan > 0:
                    _raise_if_nan(specie, "emissions")
            elif self.properties[specie]["input_mode"] == "concentration":
                n_nan = np.isnan(self.concentration.loc[dict(specie=specie)]).sum()
                if n_nan > 0:
                    _raise_if_nan(specie, "concentration")
            elif self.properties[specie]["input_mode"] == "forcing":
                n_nan = np.isnan(self.forcing.loc[dict(specie=specie)]).sum()
                if n_nan > 0:
                    _raise_if_nan(specie, "forcing")

        # same for if we are prescribing temperature; we must have non-nan
        # values in the surface level
        if self.temperature_prescribed:
            n_nan = np.isnan(self.temperature.loc[dict(layer=0)]).sum()
            if n_nan > 0:
                raise ValueError(
                    "You are running with prescribed temperatures, but the "
                    "FAIR.temperature xarray contains NaN values in the surface layer."
                )

        # special dependency cases
        if "co2" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "calculated"][
                "type"
            ]
        ):
            if "co2 ffi" not in list(
                self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                    "type"
                ]
            ) or "co2 afolu" not in list(
                self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                    "type"
                ]
            ):
                raise ValueError(
                    "co2 in calculated mode requires co2 ffi and co2 afolu in "
                    "emissions mode."
                )

        if "land use" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "calculated"][
                "type"
            ]
        ):
            if "co2 afolu" not in list(
                self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                    "type"
                ]
            ):
                raise ValueError(
                    "land use in calculated mode requires co2 afolu in emissions "
                    "mode."
                )

        if "aci" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "calculated"][
                "type"
            ]
        ):
            if self.aci_method == "stevens2015" and "sulfur" not in list(
                self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                    "type"
                ]
            ):
                raise ValueError(
                    "aci in 'calculated' mode requires sulfur in 'emissions' mode for "
                    "aci_method = stevens2015."
                )
            elif (
                self.aci_method == "smith2021"
                and "black carbon"
                not in list(
                    self.properties_df.loc[
                        self.properties_df["input_mode"] == "emissions"
                    ]["type"]
                )
                and "organic carbon"
                not in list(
                    self.properties_df.loc[
                        self.properties_df["input_mode"] == "emissions"
                    ]["type"]
                )
            ):
                raise ValueError(
                    "aci in 'calculated' mode requires sulfur, black carbon and "
                    "organic carbon in 'emissions' mode for aci_method = smith2021."
                )

        if "eesc" not in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "concentration"][
                "type"
            ]
        ) and (
            "eesc"
            in list(
                self.properties_df.loc[
                    self.properties_df["input_mode"] == "calculated"
                ]["type"]
            )
            and (
                "cfc-11"
                not in list(
                    self.properties_df.loc[
                        self.properties_df["input_mode"] == "emissions"
                    ]["type"]
                )
                and "cfc-11"
                not in list(
                    self.properties_df.loc[
                        self.properties_df["input_mode"] == "concentration"
                    ]["type"]
                )
            )
        ):
            if self.ch4_method == "thornhill2021":
                raise ValueError(
                    "For ch4_method = thornhill2021, EESC needs to be input as "
                    "concentrations, or to be calculated from emissions of "
                    "halogenated species which requires at least cfc-11 to be "
                    "provided in emissions or concentration driven mode."
                )

        co2_to_forcing = False
        ch4_to_forcing = False
        n2o_to_forcing = False

        if (
            "co2"
            in list(
                self.properties_df.loc[
                    self.properties_df["input_mode"] == "calculated"
                ]["type"]
            )
            or "co2"
            in list(
                self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                    "type"
                ]
            )
            or "co2"
            in list(
                self.properties_df.loc[
                    self.properties_df["input_mode"] == "concentration"
                ]["type"]
            )
        ):
            co2_to_forcing = True
        if "ch4" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                "type"
            ]
        ) or "ch4" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "concentration"][
                "type"
            ]
        ):
            ch4_to_forcing = True
        if "n2o" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "emissions"][
                "type"
            ]
        ) or "n2o" in list(
            self.properties_df.loc[self.properties_df["input_mode"] == "concentration"][
                "type"
            ]
        ):
            n2o_to_forcing = True
        if self.ghg_method in ["meinshausen2020", "etminan2016"]:
            if 0 < co2_to_forcing + ch4_to_forcing + n2o_to_forcing < 3:
                raise ValueError(
                    "For ghg_method in meinshausen2020, etminan2016, either all of "
                    "co2, ch4 and n2o must be provided in a form that can be "
                    "converted to concentrations, or none"
                )
        elif self.ghg_method == "myhre1998":
            if 0 < ch4_to_forcing + n2o_to_forcing < 2:
                raise ValueError(
                    "for ghg_method=myhre1998, either both of ch4 and n2o must be "
                    "provided, or neither."
                )

        for flag in [
            "ari",
            "aci",
            "ozone",
            "contrails",
            "lapsi",
            "land use",
            "h2o stratospheric",
            "eesc",
        ]:
            if flag in list(
                self.properties_df.loc[
                    self.properties_df["input_mode"] == "calculated"
                ]["type"]
            ):
                self._routine_flags[flag] = True

        # if at least one GHG is emissions, concentration or calculated from
        # precursor emissions, we want to run the forcing calculation
        if (
            (
                self.properties_df.loc[self.properties_df["greenhouse_gas"]].input_mode
                == "concentration"
            ).sum()
            + (
                self.properties_df.loc[self.properties_df["greenhouse_gas"]].input_mode
                == "emissions"
            ).sum()
            + (
                self.properties_df.loc[self.properties_df["greenhouse_gas"]].input_mode
                == "calculated"
            ).sum()
        ):
            self._routine_flags["ghg"] = True

        if self.temperature_prescribed:
            self._routine_flags["temperature"] = False

    def _make_indices(self):
        # the following are all n_species-length boolean arrays

        # these define which species do what in FaIR
        self._ghg_indices = np.asarray(
            self.properties_df.loc[:, "greenhouse_gas"].values, dtype=bool
        )
        self._co2_ffi_indices = np.asarray(
            self.properties_df["type"] == "co2 ffi", dtype=bool
        )
        self._co2_afolu_indices = np.asarray(
            self.properties_df["type"] == "co2 afolu", dtype=bool
        )
        self._co2_indices = np.asarray(self.properties_df["type"] == "co2", dtype=bool)
        self._ch4_indices = np.asarray(self.properties_df["type"] == "ch4", dtype=bool)
        self._n2o_indices = np.asarray(self.properties_df["type"] == "n2o", dtype=bool)
        self._cfc11_indices = np.asarray(
            self.properties_df["type"] == "cfc-11", dtype=bool
        )
        self._sulfur_indices = np.asarray(
            self.properties_df["type"] == "sulfur", dtype=bool
        )
        self._bc_indices = np.asarray(
            self.properties_df["type"] == "black carbon", dtype=bool
        )
        self._oc_indices = np.asarray(
            self.properties_df["type"] == "organic carbon", dtype=bool
        )
        self._aviation_nox_indices = np.asarray(
            self.properties_df["type"] == "aviation nox", dtype=bool
        )
        self._ari_indices = np.asarray(self.properties_df["type"] == "ari", dtype=bool)
        self._aci_indices = np.asarray(self.properties_df["type"] == "aci", dtype=bool)
        self._ozone_indices = np.asarray(
            self.properties_df["type"] == "ozone", dtype=bool
        )
        self._contrails_indices = np.asarray(
            self.properties_df["type"] == "contrails", dtype=bool
        )
        self._lapsi_indices = np.asarray(
            self.properties_df["type"] == "lapsi", dtype=bool
        )
        self._landuse_indices = np.asarray(
            self.properties_df["type"] == "land use", dtype=bool
        )
        self._h2ostrat_indices = np.asarray(
            self.properties_df["type"] == "h2o stratospheric", dtype=bool
        )
        self._eesc_indices = np.asarray(
            self.properties_df["type"] == "eesc", dtype=bool
        )
        self._minor_ghg_indices = (
            self._ghg_indices
            ^ self._co2_indices
            ^ self._ch4_indices
            ^ self._n2o_indices
        )
        self._halogen_indices = self._cfc11_indices | np.asarray(
            self.properties_df["type"] == "other halogen", dtype=bool
        )
        self._aerosol_chemistry_from_emissions_indices = np.asarray(
            self.properties_df.loc[:, "aerosol_chemistry_from_emissions"].values,
            dtype=bool,
        )
        self._aerosol_chemistry_from_concentration_indices = np.asarray(
            self.properties_df.loc[:, "aerosol_chemistry_from_concentration"].values,
            dtype=bool,
        )

        # and these ones are more specific, tripping certain behaviours or functions
        self._ghg_forward_indices = np.asarray(
            (
                (
                    (self.properties_df.loc[:, "input_mode"] == "emissions")
                    | (self.properties_df.loc[:, "input_mode"] == "calculated")
                )
                & (self.properties_df.loc[:, "greenhouse_gas"])
            ).values,
            dtype=bool,
        )
        self._ghg_inverse_indices = np.asarray(
            (
                (self.properties_df.loc[:, "input_mode"] == "concentration")
                & (self.properties_df.loc[:, "greenhouse_gas"])
            ).values,
            dtype=bool,
        )

    def run(self, progress=True, suppress_warnings=True):
        """Run the FaIR model.

        Parameters
        ----------
        progress : bool, optional, default=True
            Display progress bar.
        suppress_warnings : bool, optional, default=True
            Hide warnings relating to covariance in energy balance matrix.
        """
        self._check_properties()
        self._make_indices()
        if self._routine_flags["temperature"]:
            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.filterwarnings(
                        "ignore", message="covariance is not positive-semidefinite."
                    )
                self._make_ebms()

        # part of pre-run: TODO move to a new method
        if (
            self._co2_indices.sum()
            + self._co2_ffi_indices.sum()
            + self._co2_afolu_indices.sum()
            == 3
        ):
            self.emissions[..., self._co2_indices] = (
                self.emissions[..., self._co2_ffi_indices].data
                + self.emissions[..., self._co2_afolu_indices].data
            )
        self.cumulative_emissions[1:, ...] = (
            self.emissions.cumsum(axis=0, skipna=False) * self.timestep
            + self.cumulative_emissions[0, ...]
        )

        # create numpy arrays
        alpha_lifetime_array = self.alpha_lifetime.data
        airborne_emissions_array = self.airborne_emissions.data
        baseline_concentration_array = self.species_configs[
            "baseline_concentration"
        ].data
        baseline_emissions_array = self.species_configs["baseline_emissions"].data
        br_atoms_array = self.species_configs["br_atoms"].data
        ch4_lifetime_chemical_sensitivity_array = self.species_configs[
            "ch4_lifetime_chemical_sensitivity"
        ].data
        lifetime_temperature_sensitivity_array = self.species_configs[
            "lifetime_temperature_sensitivity"
        ].data
        cl_atoms_array = self.species_configs["cl_atoms"].data
        concentration_array = self.concentration.data
        concentration_per_emission_array = self.species_configs[
            "concentration_per_emission"
        ].data
        contrails_radiative_efficiency_array = self.species_configs[
            "contrails_radiative_efficiency"
        ].data
        cummins_state_array = (
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_layers + 1,
                )
            )
            * np.nan
        )
        cumulative_emissions_array = self.cumulative_emissions.data
        deep_ocean_efficacy_array = self.climate_configs["deep_ocean_efficacy"].data
        emissions_array = self.emissions.data
        erfari_radiative_efficiency_array = self.species_configs[
            "erfari_radiative_efficiency"
        ].data
        erfaci_scale_array = self.species_configs["aci_parameters"].data[:, 0]
        erfaci_shape_sulfur_array = self.species_configs["aci_parameters"].data[:, 1]
        erfaci_shape_bcoc_array = self.species_configs["aci_parameters"].data[:, 2]
        forcing_array = self.forcing.data
        forcing_scale_array = self.species_configs["forcing_scale"].data * (
            1 + self.species_configs["tropospheric_adjustment"].data
        )
        forcing_efficacy_array = self.species_configs["forcing_efficacy"].data
        forcing_efficacy_sum_array = (
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan
        )
        forcing_reference_concentration_array = self.species_configs[
            "forcing_reference_concentration"
        ].data
        forcing_sum_array = self.forcing_sum.data
        forcing_temperature_feedback_array = self.species_configs[
            "forcing_temperature_feedback"
        ].data
        fractional_release_array = self.species_configs["fractional_release"].data
        g0_array = self.species_configs["g0"].data
        g1_array = self.species_configs["g1"].data
        gas_partitions_array = np.zeros(
            (self._n_scenarios, self._n_configs, self._n_species, self._n_gasboxes)
        )
        greenhouse_gas_radiative_efficiency_array = self.species_configs[
            "greenhouse_gas_radiative_efficiency"
        ].data
        h2o_stratospheric_factor_array = self.species_configs[
            "h2o_stratospheric_factor"
        ].data
        iirf_0_array = self.species_configs["iirf_0"].data
        iirf_airborne_array = self.species_configs["iirf_airborne"].data
        iirf_temperature_array = self.species_configs["iirf_temperature"].data
        iirf_uptake_array = self.species_configs["iirf_uptake"].data
        land_use_cumulative_emissions_to_forcing_array = self.species_configs[
            "land_use_cumulative_emissions_to_forcing"
        ].data
        lapsi_radiative_efficiency_array = self.species_configs[
            "lapsi_radiative_efficiency"
        ].data
        ocean_heat_transfer_array = self.climate_configs["ocean_heat_transfer"].data
        ozone_radiative_efficiency_array = self.species_configs[
            "ozone_radiative_efficiency"
        ].data
        partition_fraction_array = self.species_configs["partition_fraction"].data
        unperturbed_lifetime_array = self.species_configs["unperturbed_lifetime"].data

        if self._routine_flags["temperature"]:
            eb_matrix_d_array = self.ebms["eb_matrix_d"].data
            forcing_vector_d_array = self.ebms["forcing_vector_d"].data
            stochastic_d_array = self.ebms["stochastic_d"].data

        # forcing should be initialised so this should not be nan. We could check, or
        # allow silent fail as some species don't take forcings and would correctly be
        # nan.
        forcing_sum_array[0:1, ...] = np.nansum(
            forcing_array[0:1, ...], axis=SPECIES_AXIS
        )

        # this is the most important state vector
        cummins_state_array[0, ..., 0] = forcing_sum_array[0, ...]
        cummins_state_array[..., 1:] = self.temperature.data

        # to save calculating this every timestep, we'll pre-determine
        # the forcing to use as the baseline value if we are using the
        # Meinshausen regime.
        if self._routine_flags["ghg"] and self.ghg_method == "meinshausen2020":
            meinshausen_baseline = meinshausen2020(
                baseline_concentration_array[None, None, ...],
                forcing_reference_concentration_array[None, None, ...],
                forcing_scale_array[None, None, ...],
                greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )

        # it's all been leading up to this : FaIR MAIN LOOP
        for i_timepoint in tqdm(
            range(self._n_timepoints),
            disable=1 - progress,
            desc=f"Running {self._n_scenarios*self._n_configs} projections in parallel",
            unit="timesteps",
        ):

            if self._routine_flags["ghg"]:
                # 1. alpha scaling
                alpha_lifetime_array[
                    i_timepoint : i_timepoint + 1, ..., self._ghg_indices
                ] = calculate_alpha(  # this timepoint
                    airborne_emissions_array[
                        i_timepoint : i_timepoint + 1, ..., self._ghg_indices
                    ],  # last timebound
                    cumulative_emissions_array[
                        i_timepoint : i_timepoint + 1, ..., self._ghg_indices
                    ],  # last timebound
                    g0_array[None, None, ..., self._ghg_indices],
                    g1_array[None, None, ..., self._ghg_indices],
                    iirf_0_array[None, None, ..., self._ghg_indices],
                    iirf_airborne_array[None, None, ..., self._ghg_indices],
                    iirf_temperature_array[None, None, ..., self._ghg_indices],
                    iirf_uptake_array[None, None, ..., self._ghg_indices],
                    cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2],
                    self.iirf_max,
                )

                # 2. multi-species methane lifetime if desired; update GHG concentration
                # for CH4
                # needs previous timebound but this is no different to the generic
                if self.ch4_method == "thornhill2021":
                    alpha_lifetime_array[
                        i_timepoint : i_timepoint + 1, ..., self._ch4_indices
                    ] = calculate_alpha_ch4(
                        emissions_array[i_timepoint : i_timepoint + 1, ...],
                        concentration_array[i_timepoint : i_timepoint + 1, ...],
                        cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2],
                        baseline_emissions_array[None, None, ...],
                        baseline_concentration_array[None, None, ...],
                        ch4_lifetime_chemical_sensitivity_array[None, None, ...],
                        lifetime_temperature_sensitivity_array[None, None, :, None],
                        self._aerosol_chemistry_from_emissions_indices,
                        self._aerosol_chemistry_from_concentration_indices,
                    )

                # 3. greenhouse emissions to concentrations; include methane from IIRF
                (
                    concentration_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        self._ghg_forward_indices,
                    ],
                    gas_partitions_array[..., self._ghg_forward_indices, :],
                    airborne_emissions_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        self._ghg_forward_indices,
                    ],
                ) = step_concentration(
                    emissions_array[
                        i_timepoint : i_timepoint + 1,
                        ...,
                        self._ghg_forward_indices,
                        None,
                    ],  # this timepoint
                    gas_partitions_array[
                        ..., self._ghg_forward_indices, :
                    ],  # last timebound
                    airborne_emissions_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        self._ghg_forward_indices,
                        None,
                    ],  # last timebound
                    alpha_lifetime_array[
                        i_timepoint : i_timepoint + 1,
                        ...,
                        self._ghg_forward_indices,
                        None,
                    ],
                    baseline_concentration_array[
                        None, None, ..., self._ghg_forward_indices
                    ],
                    baseline_emissions_array[
                        None, None, ..., self._ghg_forward_indices, None
                    ],
                    concentration_per_emission_array[
                        None, None, ..., self._ghg_forward_indices
                    ],
                    unperturbed_lifetime_array[
                        None, None, ..., self._ghg_forward_indices, :
                    ],
                    #        oxidation_matrix,
                    partition_fraction_array[
                        None, None, ..., self._ghg_forward_indices, :
                    ],
                    self.timestep,
                )

                # 4. greenhouse gas concentrations to emissions
                (
                    emissions_array[
                        i_timepoint : i_timepoint + 1, ..., self._ghg_inverse_indices
                    ],
                    gas_partitions_array[..., self._ghg_inverse_indices, :],
                    airborne_emissions_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        self._ghg_inverse_indices,
                    ],
                ) = unstep_concentration(
                    concentration_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        self._ghg_inverse_indices,
                    ],  # this timepoint
                    gas_partitions_array[
                        None, ..., self._ghg_inverse_indices, :
                    ],  # last timebound
                    airborne_emissions_array[
                        i_timepoint : i_timepoint + 1,
                        ...,
                        self._ghg_inverse_indices,
                        None,
                    ],  # last timebound
                    alpha_lifetime_array[
                        i_timepoint : i_timepoint + 1,
                        ...,
                        self._ghg_inverse_indices,
                        None,
                    ],
                    baseline_concentration_array[
                        None, None, ..., self._ghg_inverse_indices
                    ],
                    baseline_emissions_array[
                        None, None, ..., self._ghg_inverse_indices
                    ],
                    concentration_per_emission_array[
                        None, None, ..., self._ghg_inverse_indices
                    ],
                    unperturbed_lifetime_array[
                        None, None, ..., self._ghg_inverse_indices, :
                    ],
                    #        oxidation_matrix,
                    partition_fraction_array[
                        None, None, ..., self._ghg_inverse_indices, :
                    ],
                    self.timestep,
                )
                cumulative_emissions_array[
                    i_timepoint + 1, ..., self._ghg_inverse_indices
                ] = (
                    cumulative_emissions_array[
                        i_timepoint, ..., self._ghg_inverse_indices
                    ]
                    + emissions_array[i_timepoint, ..., self._ghg_inverse_indices]
                    * self.timestep
                )

                # 5. greenhouse gas concentrations to forcing
                if self.ghg_method == "leach2021":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                    ] = leach2021ghg(
                        concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                        baseline_concentration_array[None, None, ...]
                        * np.ones(
                            (1, self._n_scenarios, self._n_configs, self._n_species)
                        ),
                        forcing_scale_array[None, None, ...],
                        greenhouse_gas_radiative_efficiency_array[None, None, ...],
                        self._co2_indices,
                        self._ch4_indices,
                        self._n2o_indices,
                        self._minor_ghg_indices,
                    )[
                        0:1, ..., self._ghg_indices
                    ]
                if self.ghg_method == "meinshausen2020":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                    ] = meinshausen2020(
                        concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                        forcing_reference_concentration_array[None, None, ...]
                        * np.ones(
                            (1, self._n_scenarios, self._n_configs, self._n_species)
                        ),
                        forcing_scale_array[None, None, ...],
                        greenhouse_gas_radiative_efficiency_array[None, None, ...],
                        self._co2_indices,
                        self._ch4_indices,
                        self._n2o_indices,
                        self._minor_ghg_indices,
                    )[
                        0:1, ..., self._ghg_indices
                    ]
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                    ] = (
                        forcing_array[
                            i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                        ]
                        - meinshausen_baseline[..., self._ghg_indices]
                    )
                elif self.ghg_method == "etminan2016":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                    ] = etminan2016(
                        concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                        baseline_concentration_array[None, None, ...]
                        * np.ones(
                            (1, self._n_scenarios, self._n_configs, self._n_species)
                        ),
                        forcing_scale_array[None, None, ...],
                        greenhouse_gas_radiative_efficiency_array[None, None, ...],
                        self._co2_indices,
                        self._ch4_indices,
                        self._n2o_indices,
                        self._minor_ghg_indices,
                    )[
                        0:1, ..., self._ghg_indices
                    ]
                elif self.ghg_method == "myhre1998":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                    ] = myhre1998(
                        concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                        baseline_concentration_array[None, None, ...]
                        * np.ones(
                            (1, self._n_scenarios, self._n_configs, self._n_species)
                        ),
                        forcing_scale_array[None, None, ...],
                        greenhouse_gas_radiative_efficiency_array[None, None, ...],
                        self._co2_indices,
                        self._ch4_indices,
                        self._n2o_indices,
                        self._minor_ghg_indices,
                    )[
                        0:1, ..., self._ghg_indices
                    ]

            # 6. aerosol direct forcing
            if self._routine_flags["ari"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._ari_indices
                ] = calculate_erfari_forcing(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_emissions_array[None, None, ...],
                    baseline_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ...],
                    erfari_radiative_efficiency_array[None, None, ...],
                    self._aerosol_chemistry_from_emissions_indices,
                    self._aerosol_chemistry_from_concentration_indices,
                )

            # 7. aerosol indirect forcing
            if self._routine_flags["aci"]:
                if self.aci_method == "stevens2015":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._aci_indices
                    ] = stevens2015(
                        emissions_array[i_timepoint : i_timepoint + 1, ...],
                        baseline_emissions_array[None, None, ...],
                        forcing_scale_array[None, None, ..., self._aci_indices],
                        erfaci_scale_array[None, None, :, None],
                        erfaci_shape_sulfur_array[None, None, :, None],
                        self._sulfur_indices,
                    )
                elif self.aci_method == "smith2021":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._aci_indices
                    ] = smith2021(
                        emissions_array[i_timepoint : i_timepoint + 1, ...],
                        baseline_emissions_array[None, None, ...],
                        forcing_scale_array[None, None, ..., self._aci_indices],
                        erfaci_scale_array[None, None, :, None],
                        erfaci_shape_sulfur_array[None, None, :, None],
                        erfaci_shape_bcoc_array[None, None, :, None],
                        self._sulfur_indices,
                        self._bc_indices,
                        self._oc_indices,
                    )
                elif self.aci_method == "leach2021":
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., self._aci_indices
                    ] = leach2021aci(
                        emissions_array[i_timepoint : i_timepoint + 1, ...],
                        baseline_emissions_array[None, None, ...],
                        forcing_scale_array[None, None, ..., self._aci_indices],
                        erfaci_scale_array[None, None, :, None],
                        erfaci_shape_sulfur_array[None, None, :, None],
                        erfaci_shape_bcoc_array[None, None, :, None],
                        self._sulfur_indices,
                        self._bc_indices,
                        self._oc_indices,
                    )

            # 8. calculate EESC this timestep for ozone forcing (and use it for
            # methane lifetime in the following timestep)
            if self._routine_flags["eesc"]:
                concentration_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._eesc_indices
                ] = calculate_eesc(
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_concentration_array[None, ...],
                    fractional_release_array[None, None, ...],
                    cl_atoms_array[None, None, ...],
                    br_atoms_array[None, None, ...],
                    self._cfc11_indices,
                    self._halogen_indices,
                    self.br_cl_ods_potential,
                )

            # 9. ozone emissions & concentrations to forcing
            if self._routine_flags["ozone"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._ozone_indices
                ] = thornhill2021(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_emissions_array[None, None, ...],
                    baseline_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ..., self._ozone_indices],
                    ozone_radiative_efficiency_array[None, None, ...],
                    self._aerosol_chemistry_from_emissions_indices,
                    self._aerosol_chemistry_from_concentration_indices,
                )

            # 10. contrails forcing from NOx emissions
            if self._routine_flags["contrails"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._contrails_indices
                ] = calculate_linear_forcing(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    baseline_emissions_array[None, None, ...],
                    forcing_scale_array[None, None, ..., self._contrails_indices],
                    contrails_radiative_efficiency_array[None, None, ...],
                )

            # 11. LAPSI forcing from BC and OC emissions
            if self._routine_flags["lapsi"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._lapsi_indices
                ] = calculate_linear_forcing(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    baseline_emissions_array[None, None, ...],
                    forcing_scale_array[None, None, ..., self._lapsi_indices],
                    lapsi_radiative_efficiency_array[None, None, ...],
                )

            # 12. CH4 forcing to stratospheric water vapour forcing
            if self._routine_flags["h2o stratospheric"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._h2ostrat_indices
                ] = calculate_linear_forcing(
                    forcing_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    forcing_array[0:1, ...],
                    forcing_scale_array[None, None, ..., self._h2ostrat_indices],
                    h2o_stratospheric_factor_array[None, None, ...],
                )

            # 13. CO2 cumulative emissions to land use change forcing
            if self._routine_flags["land use"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._landuse_indices
                ] = calculate_linear_forcing(
                    cumulative_emissions_array[i_timepoint : i_timepoint + 1, ...],
                    cumulative_emissions_array[0:1, ...],
                    forcing_scale_array[None, None, ..., self._landuse_indices],
                    land_use_cumulative_emissions_to_forcing_array[None, None, ...],
                )

            # 14. apply temperature-forcing feedback here.
            forcing_array[i_timepoint + 1 : i_timepoint + 2, ...] = (
                forcing_array[i_timepoint + 1 : i_timepoint + 2, ...]
                + cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2]
                * forcing_temperature_feedback_array[None, None, ...]
            )

            # 15. sum forcings
            forcing_sum_array[i_timepoint + 1 : i_timepoint + 2, ...] = np.nansum(
                forcing_array[i_timepoint + 1 : i_timepoint + 2, ...], axis=SPECIES_AXIS
            )
            forcing_efficacy_sum_array[
                i_timepoint + 1 : i_timepoint + 2, ...
            ] = np.nansum(
                forcing_array[i_timepoint + 1 : i_timepoint + 2, ...]
                * forcing_efficacy_array[None, None, ...],
                axis=SPECIES_AXIS,
            )

            # 16. forcing to temperature
            if self._routine_flags["temperature"]:
                cummins_state_array[
                    i_timepoint + 1 : i_timepoint + 2, ...
                ] = step_temperature(
                    cummins_state_array[i_timepoint : i_timepoint + 1, ...],
                    eb_matrix_d_array[None, None, ...],
                    forcing_vector_d_array[None, None, ...],
                    stochastic_d_array[i_timepoint + 1 : i_timepoint + 2, None, ...],
                    forcing_efficacy_sum_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., None
                    ],
                )

        # 17. TOA imbalance
        # forcing is not efficacy adjusted here, is this correct?
        toa_imbalance_array = calculate_toa_imbalance_postrun(
            cummins_state_array,
            forcing_sum_array,  # [..., None],
            ocean_heat_transfer_array,
            deep_ocean_efficacy_array,
        )

        # 18. Ocean heat content change
        ocean_heat_content_change_array = (
            np.cumsum(toa_imbalance_array * self.timestep, axis=TIME_AXIS)
            * earth_radius**2
            * 4
            * np.pi
            * seconds_per_year
        )

        # 19. calculate airborne fraction - we have NaNs and zeros we know about, and we
        # don't mind
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            airborne_fraction_array = (
                airborne_emissions_array / cumulative_emissions_array
            )

        # 20. (Re)allocate to xarray
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

    def to_netcdf(self, filename):
        """Write out FaIR scenario data to a netCDF file.

        Parameters
        ----------
        filename : str
            file path of the file to write.
        """
        ds = xr.Dataset(
            data_vars=dict(
                emissions=(
                    ["timepoint", "scenario", "config", "specie"],
                    self.emissions.data,
                ),
                concentration=(
                    ["timebound", "scenario", "config", "specie"],
                    self.concentration.data,
                ),
                forcing=(
                    ["timebound", "scenario", "config", "specie"],
                    self.forcing.data,
                ),
                forcing_sum=(
                    ["timebound", "scenario", "config"],
                    self.forcing_sum.data,
                ),
                temperature=(
                    ["timebound", "scenario", "config", "layer"],
                    self.temperature.data,
                ),
                airborne_emissions=(
                    ["timebound", "scenario", "config", "specie"],
                    self.airborne_emissions.data,
                ),
                airborne_fraction=(
                    ["timebound", "scenario", "config", "specie"],
                    self.airborne_fraction.data,
                ),
                cumulative_emissions=(
                    ["timebound", "scenario", "config", "specie"],
                    self.cumulative_emissions.data,
                ),
                ocean_heat_content_change=(
                    ["timebound", "scenario", "config"],
                    self.ocean_heat_content_change.data,
                ),
                stochastic_forcing=(
                    ["timebound", "scenario", "config"],
                    self.stochastic_forcing.data,
                ),
                toa_imbalance=(
                    ["timebound", "scenario", "config"],
                    self.toa_imbalance.data,
                ),
            ),
            coords=dict(
                timepoint=self.timepoints,
                timebound=self.timebounds,
                scenario=self.scenarios,
                config=self.configs,
                specie=self.species,
                layer=self.layers,
            ),
        )
        ds.to_netcdf(filename)
