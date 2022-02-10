from dataclasses import dataclass, field
from enum import Enum, auto
from ..exceptions import InvalidRunModeError

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
    SLCF_OZONE_PRECURSOR = auto()
    AVIATION_NOX = auto()
    OZONE = auto()
    AEROSOL_CLOUD_INTERACTIONS = auto()
    CONTRAILS = auto()
    LAPSI = auto()
    H2O_STRATOSPHERIC = auto()
    LAND_USE = auto()
    VOLCANIC = auto()
    SOLAR = auto()


class AggregatedCategory():
    """Groups of Category that share common properties."""
    CO2_SOURCE = [Category.CO2_FFI, Category.CO2_AFOLU] 
    GREENHOUSE_GAS = [Category.CO2, Category.CH4, Category.N2O, Category.CFC_11, Category.OTHER_HALOGEN, Category.F_GAS]
    HALOGEN = [Category.CFC_11, Category.OTHER_HALOGEN]
    AEROSOL = [Category.SULFUR, Category.BC, Category.OC, Category.OTHER_AEROSOL]
    OZONE_PRECURSOR = [Category.CH4, Category.N2O, Category.SLCF_OZONE_PRECURSOR] + HALOGEN
    NO_DUPLICATES_ALLOWED = [
        Category.CO2_FFI,
        Category.CO2_AFOLU,
        Category.CO2,
        Category.CH4,
        Category.N2O,
        Category.CFC_11,
        Category.SULFUR,
        Category.BC,
        Category.OC,
        Category.AVIATION_NOX,
        Category.OZONE,
        Category.AEROSOL_CLOUD_INTERACTIONS,
        Category.CONTRAILS,
        Category.LAPSI,
        Category.H2O_STRATOSPHERIC,
        Category.LAND_USE,
        Category.SOLAR,
        Category.VOLCANIC
    ]

class RunMode(Enum):
    EMISSIONS = auto()
    CONCENTRATION = auto()
    FROM_OTHER_SPECIES = auto()
    FORCING = auto()


# move these?
valid_run_modes = {
    Category.CO2_FFI: (RunMode.EMISSIONS,),
    Category.CO2_AFOLU: (RunMode.EMISSIONS,),
    Category.CO2: (RunMode.CONCENTRATION, RunMode.FORCING, RunMode.FROM_OTHER_SPECIES),  # we will only allow CO2 emissions to be given for FFI and AFOLU separately.
    Category.CH4: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.N2O: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.CFC_11: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.OTHER_HALOGEN: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.F_GAS: (RunMode.EMISSIONS, RunMode.CONCENTRATION, RunMode.FORCING),
    Category.SULFUR: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.BC: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.OC: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.OTHER_AEROSOL: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.SLCF_OZONE_PRECURSOR: (RunMode.EMISSIONS, RunMode.FORCING),
    Category.AVIATION_NOX: (RunMode.EMISSIONS,),
    Category.OZONE: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.AEROSOL_CLOUD_INTERACTIONS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.CONTRAILS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.LAPSI: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.H2O_STRATOSPHERIC: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.LAND_USE: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.SOLAR: (RunMode.FORCING,),
    Category.VOLCANIC: (RunMode.FORCING,),
}


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
            elif self.category in [
                Category.OZONE,
                Category.AEROSOL_CLOUD_INTERACTIONS,
                Category.CONTRAILS,
                Category.H2O_STRATOSPHERIC,
                Category.LAPSI,
                Category.LAND_USE
            ]:
                self.run_mode = RunMode.FROM_OTHER_SPECIES
            else:
                self.run_mode = RunMode.EMISSIONS
        # 2. check valid run mode for each species given
        if self.run_mode not in valid_run_modes[self.category]:
            raise InvalidRunModeError(f"cannot run {self.category} in {self.run_mode} mode")


class ACIMethod(Enum):
    STEVENS2015 = auto()
    SMITH2018 = auto()


@dataclass
class RunConfig():
    n_gas_boxes: int=4
    n_temperature_boxes: int=3
    temperature_prescribed: bool=False
    aci_method: ACIMethod=ACIMethod.SMITH2018
    br_cl_ratio: float=45
    iirf_horizon: float=100
    iirf_max: float=99.95
