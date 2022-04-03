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
    NOX_AVIATION = auto()
    OZONE = auto()
    AEROSOL_RADIATION_INTERACTIONS = auto()
    AEROSOL_CLOUD_INTERACTIONS = auto()
    CONTRAILS = auto()
    LAPSI = auto()
    H2O_STRATOSPHERIC = auto()
    LAND_USE = auto()
    VOLCANIC = auto()
    SOLAR = auto()
    OTHER = auto()


class AggregatedCategory():
    """Groups of Category that share common properties."""
    CO2_SOURCE = [Category.CO2_FFI, Category.CO2_AFOLU]
    GREENHOUSE_GAS = [Category.CO2, Category.CH4, Category.N2O, Category.CFC_11, Category.OTHER_HALOGEN, Category.F_GAS]
    MINOR_GREENHOUSE_GAS = [Category.CFC_11, Category.OTHER_HALOGEN, Category.F_GAS]
    HALOGEN = [Category.CFC_11, Category.OTHER_HALOGEN]
    AEROSOL = [Category.SULFUR, Category.BC, Category.OC, Category.OTHER_AEROSOL]
    OZONE_PRECURSOR = [Category.CH4, Category.N2O, Category.SLCF_OZONE_PRECURSOR] + HALOGEN
    SLCF = AEROSOL + [Category.SLCF_OZONE_PRECURSOR]
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
        Category.NOX_AVIATION,
        Category.OZONE,
        Category.AEROSOL_RADIATION_INTERACTIONS,
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
    Category.SULFUR: (RunMode.EMISSIONS,),
    Category.BC: (RunMode.EMISSIONS,),
    Category.OC: (RunMode.EMISSIONS,),
    Category.OTHER_AEROSOL: (RunMode.EMISSIONS,),
    Category.SLCF_OZONE_PRECURSOR: (RunMode.EMISSIONS,),
    Category.NOX_AVIATION: (RunMode.EMISSIONS,),
    Category.OZONE: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.AEROSOL_RADIATION_INTERACTIONS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.AEROSOL_CLOUD_INTERACTIONS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.CONTRAILS: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.LAPSI: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.H2O_STRATOSPHERIC: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.LAND_USE: (RunMode.FROM_OTHER_SPECIES, RunMode.FORCING),
    Category.SOLAR: (RunMode.FORCING,),
    Category.VOLCANIC: (RunMode.FORCING,),
    Category.OTHER: (RunMode.FORCING,),
}


@dataclass
class SpeciesID():
    """Defines basic properties of each Species included in the model.

    Attributes
    ----------
    name : str
        A unique name to define this SpeciesID, for example the name of the
        greenhouse gas or aerosol Species being represented.
    category : Category
        The type of Species being represented.
    run_mode : RunMode, optional
        How the species is being introduced in the model (emissions-driven,
        concentration-driven, forcing-driven, or calculated from other Species).
        If not provided, a default RunMode is selected based on the Category.
    """

    name: str
    category: Category
    run_mode: RunMode=None

    def __post_init__(self):
        # 1. fill default run_mode
        if self.run_mode is None:
            if self.category in [Category.SOLAR, Category.VOLCANIC, Category.OTHER]:
                self.run_mode = RunMode.FORCING
            elif self.category in [
                Category.CO2,
                Category.OZONE,
                Category.AEROSOL_RADIATION_INTERACTIONS,
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
    """Defines the aerosol-cloud interactions relationship to use.

    Attributes
    ----------
    STEVENS2015 : enum
        Use relationship based on sulfur emissions from Stevens (2015) [1]_.
    SMITH2018 : enum
        Use relationship based on sulfur, organic carbon and black carbon
        emissions based on FaIRv1.3 (Smith et al. 2018 [2]_).


    References
    ----------
    .. [1] Stevens, B. (2015). Rethinking the Lower Bound on Aerosol Radiative
        Forcing, Journal of Climate, 28(12), 4794-4819.

    .. [2] Smith, C. J., Forster, P. M.,  Allen, M., Leach, N., Millar, R. J.,
        Passerello, G. A., and Regayre, L. A. (2018). FAIR v1.3: a simple
        emissions-based impulse response and carbon cycle model, Geosci. Model
        Dev., 11, 2273–2297
    """
    STEVENS2015 = auto()
    SMITH2018 = auto()


class CH4LifetimeMethod(Enum):
    """Defines the methane lifetime method to use.

    Attributes
    ----------
    LEACH2021 : enum
        Use Leach et al. (2021) FaIR v2.0 treatment which is the same as for
        other GHGs (methane lifetime depends only on methane and climate)
    THORNHILL2021 : enum
        Use Thornhill et al. (2021a, 2021b) treatment of various precursor gases
        and SLCFs and climate affecting methane lifetime, based on calibrations
        to AerChemMIP CMIP6 models.
    """
    LEACH2021 = auto()
    THORNHILL2021 = auto()


@dataclass
class RunConfig():
    n_gas_boxes: int=4
    n_temperature_boxes: int=3
    temperature_prescribed: bool=False
    aci_method: ACIMethod=ACIMethod.SMITH2018
    ch4_lifetime_method=CH4LifetimeMethod.LEACH2021
    br_cl_ratio: float=45
    iirf_horizon: float=100
    iirf_max: float=99.5
