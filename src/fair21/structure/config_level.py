from dataclasses import dataclass, field
from numbers import Number
import typing
from typing import Iterable  # sort this out

import numpy as np

from .top_level import SpeciesID, Category, AggregatedCategory, RunConfig
from ..earth_params import mass_atmosphere, molecular_weight_air

@dataclass
class ClimateResponse():
    """Defines how the climate responds to forcing.

    A broader description is provided in `fair.energy_balance_model`.

    Attributes
    ----------
        ocean_heat_capacity : float or array_like
            Ocean heat capacity of each layer (top first), W m-2 yr K-1
        ocean_heat_transfer : float or array_like
            Heat exchange coefficient between ocean layers (top first). The
            first element of this array is akin to the climate feedback
            parameter, with the convention that stabilising feedbacks are
            positive (opposite to most climate sensitivity literature).
            W m-2 K-1
        deep_ocean_efficacy : float
            efficacy of deepest ocean layer. See e.g. [1]_.
        stochastic_run : bool
            Activate the stochastic variability component from [2]_.
        sigma_eta : float
            Standard deviation of stochastic forcing component from [2]_.
        sigma_xi : float
            Standard deviation of stochastic disturbance applied to surface
            layer. See [2]_.
        gamma_autocorrelation : float
            Stochastic forcing continuous-time autocorrelation parameter.
            See [2]_.
        seed : int or None
            Random seed to use for stochastic variability.

    Raises
    ------
    WrongArrayShapeError
        if ocean_heat_capacity and ocean_heat_transfer are not at most 1D
    IncompatibleConfigError
        if ocean_heat_capacity and ocean_heat_transfer are different shapes
    TypeError
        if deep_ocean_efficacy or any stochasic parameters when
        stochastic_run=True are not positive numbers.

    References
    ----------
    .. [1] Geoffroy, O., Saint-Martin, D., Bellon, G., Voldoire, A., Olivié,
        D. J. L., & Tytéca, S. (2013). Transient Climate Response in a Two-
        Layer Energy-Balance Model. Part II: Representation of the Efficacy
        of Deep-Ocean Heat Uptake and Validation for CMIP5 AOGCMs, Journal
        of Climate, 26(6), 1859-1876

    .. [2] Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
        Estimation of Stochastic Energy Balance Model Parameters, Journal of
        Climate, 33(18), 7909-7926.
    """
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
    contrails_emissions_to_forcing: float=0
    lapsi_emissions_to_forcing: float=0
    h2o_stratospheric_factor: float=0
    land_use_cumulative_emissions_to_forcing: float=0
    baseline_emissions: float=0
    ozone_radiative_efficiency: float=None
    cl_atoms: int=0
    br_atoms: int=0
    fractional_release: float=None
    tropospheric_adjustment: float=0
    scale: float=1
    efficacy: float=1
    aci_params: dict=None
    forcing_temperature_feedback: float=0
    run_config: RunConfig=RunConfig()

    def __post_init__(self):
        # validate input - the whole partition_fraction and lifetime thing
        # would be nice to validate if not CO2, CH4 or N2O that radiative_efficiency must be defined.
#        if self.partition_fraction is not isinstance(Number, Iterable):
#            raise MissingInputError('partition_fraction must be a number or an array-like type') # custom exception needed
#                    if len(partition_fraction) != len(lifetime):
#                        raise IncompatibleConfigError('`partition_fraction` and `lifetime` are different shapes') # custom exception needed
#                    if ~np.isclose(np.sum(partition_fraction), 1):
#                        raise PartitionFractionError('partition_fraction should sum to 1') # custom exception needed
#        elif np.ndim(self.lifetime) > 1:
#            raise LifetimeError('`lifetime` array dimension is greater than 1')
        if self.species_id.category in AggregatedCategory.GREENHOUSE_GAS:
            self.g1 = np.sum(
                np.asarray(self.partition_fraction) * np.asarray(self.lifetime) *
                (1 - (1 + self.run_config.iirf_horizon/np.asarray(self.lifetime)) *
                np.exp(-self.run_config.iirf_horizon/np.asarray(self.lifetime)))
            )
            self.g0 = np.exp(-1 * np.sum(np.asarray(self.partition_fraction)*
                np.asarray(self.lifetime)*
                (1 - np.exp(-self.run_config.iirf_horizon/np.asarray(self.lifetime))))/
                self.g1
            )
            self.burden_per_emission = 1 / (mass_atmosphere / 1e18 * self.molecular_weight / molecular_weight_air)
            if self.iirf_0 is None:
                self.iirf_0 = (
                    np.sum(np.asarray(self.lifetime) *
                    (1 - np.exp(-self.run_config.iirf_horizon / np.asarray(self.lifetime)))
                    * np.asarray(self.partition_fraction))
                )

        if self.species_id.category in AggregatedCategory.HALOGEN:
            if not isinstance(self.ozone_radiative_efficiency, Number):
                raise ValueError("ozone_properties.ozone_radiative_efficiency should be a number for Halogens")
            if not isinstance(self.cl_atoms, int) or self.cl_atoms < 0:
                raise ValueError("ozone_properties.cl_atoms should be a non-negative integer for Halogens")
            if not isinstance(self.br_atoms, int) or self.cl_atoms < 0:
                raise ValueError("ozone_properties.br_atoms should be a non-negative integer for Halogens")
            if not isinstance(self.fractional_release, Number) or self.fractional_release < 0:
                raise ValueError("ozone_properties.fractional_release should be a non-negative number for Halogens")

        if self.species_id.category == Category.AEROSOL_CLOUD_INTERACTIONS:
            if not isinstance(self.aci_params, dict):
                raise TypeError("For aerosol-cloud interactions, you must supply a dict of parameters using the aci_params keyword")

        if self.species_id.category == Category.AVIATION_NOX:
            if not isinstance(self.contrails_emissions_to_forcing, Number):
                raise ValueError("contrails_emissions_to_forcing should be a number")

        if not isinstance(self.forcing_temperature_feedback, Number):
            raise ValueError("forcing_temperature_feedback should be a number")


@dataclass
class Config():
    name: str
    climate_response: ClimateResponse
    species_configs: typing.List[SpeciesConfig]

    def __post_init__(self):
        # check eveything provided is a Config
        if not isinstance(self.species_configs, list):
            raise TypeError('species_configs argument passed to Config must be a list of SpeciesConfig')
