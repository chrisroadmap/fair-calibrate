
from . import _version
__version__ = _version.get_versions()['version']

# we want isort to not alphabetise these, because they correspond to different
# levels or array indices in the FaIR ecosystem
from .structure.top_level import SpeciesID, Category, RunMode, ACIMethod, CH4LifetimeMethod, RunConfig
from .structure.scenario_level import Scenario
from .structure.config_level import ClimateResponse, SpeciesConfig, Config
from .structure.species_level import Species

from .fair import FAIR
