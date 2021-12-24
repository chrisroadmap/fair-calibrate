from dataclasses import dataclass, field
import typing

import numpy as np


@dataclass
class Species():
    name: str
    tropospheric_adjustment: float=0
    scale: float=1


@dataclass
class Emissions():
    species: Species
    emissions: typing.Union[tuple, list, np.ndarray]
    baseline: float = 0
    natural_emissions_adjustment: float=0


@dataclass
class Concentration():
    species: Species
    concentration: typing.Union[tuple, list, np.ndarray]
    baseline: float=0


@dataclass
class Forcing():
    species: Species
    forcing: typing.Union[tuple, list, np.ndarray]


#class IIRF_0():



@dataclass
class IIRF():
    iirf_0: float=field(default=None)
    iirf_airborne: float=0
    iirf_cumulative: float=0
    iirf_temperature: float=0
    iirf_horizon: float=100
    lifetime: float=field(default=None, repr=False)
    partition_fraction: typing.Union[float, tuple, list, np.ndarray] = field(default=1, repr=False)

    def __post_init__(self):
        if self.iirf_0 is None:
            self.iirf_0 = (
                np.sum(np.asarray(self.lifetime) *
                (1 - np.exp(-self.iirf_horizon / np.asarray(self.lifetime)))
                * np.asarray(self.partition_fraction)
            )

iirf = IIRF(29, 0, 0.003, 4)
print(iirf)

iirf2 = IIRF(lifetime=27, partition_fraction=(0.3, 0.2, 0.1, 0.4))
print(iirf2)

iirf_co2 = IIRF(lifetime=[1e9, 394, 36, 4], partition_fraction=[0.2173,0.2240,0.2824,0.2763])
print(iirf_co2)

iirf_co2 = IIRF(29, 0.000819, 0.00846, 4)
print(iirf_co2)

@dataclass
class GreenhouseGas():
    species: Species
    molecular_weight: float
    radiative_effiency: float
    lifetime: typing.Union[float, tuple, list, np.ndarray]
    partition_fraction: typing.Union[float, tuple, list, np.ndarray]
    iirf: IIRF
    g0: float=field(init=False)
    g1: float=field(init=False)

    def __post_init__(self):
        self.g1 = np.sum(
            np.asarray(self.partition_fraction) * np.asarray(self.lifetime) *
            (1 - (1 + self.iirf.iirf_horizon/np.asarray(self.lifetime)) *
            np.exp(-self.iirf.iirf_horizon/np.asarray(self.lifetime)))
        )
        self.g0 = np.exp(-1 * np.sum(np.asarray(self.partition_fraction)*
            np.asarray(self.lifetime)*
            (1 - np.exp(-self.iirf.iirf_horizon/np.asarray(self.lifetime))))/
            self.g1
        )

        """Validator for attributes provided to GreenhouseGas"""


    # def __init__(self, name, molecular_weight, lifetime, radiative_efficiency, **kwargs):
    #     tropospheric_adjustment = kwargs.pop('tropospheric_adjustment', 0)
    #     super().__init__(name, tropospheric_adjustment)
    #
    #     # move the below to input verification method
    #     self.molecular_weight = molecular_weight
    #     self.lifetime = lifetime
    #     self.radiative_efficiency = radiative_efficiency
    #     if np.ndim(self.lifetime) == 1:
    #         lifetime = np.asarray(lifetime)
    #         # should we enforce whether strictly decreasing or not?
    #         partition_fraction = kwargs.get('partition_fraction')
    #         if partition_fraction is None:
    #             raise MissingInputError('specify `partition_fraction` if specifying more than one `lifetime`')
     # custom exception needed
    #         if len(partition_fraction) != len(lifetime):
    #             raise IncompatibleConfigError('`partition_fraction` and `lifetime` are different shapes')
    # custom exception needed
    #         partition_fraction = np.asarray(partition_fraction)
    #         if ~np.isclose(np.sum(partition_fraction), 1):
    #             raise PartitionFractionError('partition_fraction should sum to 1') # custom exception needed
    #     elif np.ndim(self.lifetime) > 1:
    #         raise LifetimeError('`lifetime` array dimension is greater than 1')
    #     else:
    #         partition_fraction=np.zeros(n_gas_boxes)
    #         partition_fraction[0] = 1
    #     self.partition_fraction=partition_fraction
    #     self.emissions = kwargs.get('emissions')
    #     self.concentration = kwargs.get('concentration')
    #     self.forcing = kwargs.get('forcing')
    #
    #     self.iirf_0 = kwargs.get('iirf_0', self.lifetime_to_iirf_0(lifetime,
    #partition_fraction, iirf_horizon=iirf_horizon))
    #     self.iirf_cumulative = kwargs.get('iirf_cumulative', 0)
    #     self.iirf_airborne = kwargs.get('iirf_airborne', 0)
    #     self.iirf_temperature = kwargs.get('iirf_temperature', 0)
    #     self.pre_industrial_emissions=kwargs.get('pre_industrial_emissions', 0)
    #     self.pre_industrial_concentration=kwargs.get('pre_industrial_concentration', 0)
    #     self.natural_emissions_adjustment=kwargs.get('natural_emissions_adjustment', 0)
    #     self.output_forcing=True

print()
print()
co2 = GreenhouseGas(
    Species("CO2"),
    44.009,
    1.33e-05,
    (1e9,394.4,36.54,4.304),
    (0.2173,0.2240,0.2824,0.2763),
    iirf_co2
)
print(co2)

print()
print()

co2_emissions = Emissions(Species('CO2'), np.arange(10), 0)
print(co2_emissions)



### main
for i in range(751):
    
