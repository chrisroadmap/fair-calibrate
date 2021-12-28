from dataclasses import dataclass, field
import typing

import numpy as np


@dataclass
class Species():
    name: str


@dataclass
class Emissions():
    species: Species
    emissions: typing.Union[tuple, list, np.ndarray]
    baseline: float = 0
    natural_emissions_adjustment: float=0

    def __post_init__(self):
        if not isinstance(self.species, Species):
            raise ValueError(f"{self.species} is not of type Species")


@dataclass
class Concentration():
    species: Species
    concentration: typing.Union[tuple, list, np.ndarray]
    baseline: float=0


@dataclass
class Forcing():
    species: Species
    forcing: typing.Union[tuple, list, np.ndarray]
    tropospheric_adjustment: float=0
    scale: float=1


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
                np.sum(np.asarray(self.lifetime)) *
                (1 - np.exp(-self.iirf_horizon / np.asarray(self.lifetime)))
                * np.asarray(self.partition_fraction)
            )

co2_iirf = IIRF(lifetime=[1e9, 394, 36, 4], partition_fraction=[0.2173,0.2240,0.2824,0.2763])
print(co2_iirf)

co2_iirf = IIRF(29, 0.000819, 0.00846, 4)
print(co2_iirf)

@dataclass
class GreenhouseGas():
    species: Species
    molecular_weight: float
    radiative_efficiency: float
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

print()
print()
co2_species = Species("CO2")

co2_config = GreenhouseGas(
    co2_species,
    molecular_weight=44.009,
    radiative_efficiency=1.33e-05,
    lifetime=(1e9,394.4,36.54,4.304),
    partition_fraction=(0.2173,0.2240,0.2824,0.2763),
    iirf=co2_iirf
)
print(co2_config)

print()
print()

co2_emissions = Emissions(co2_species, np.arange(10), 0)
print(co2_emissions)



### main
#for i in range(751):
