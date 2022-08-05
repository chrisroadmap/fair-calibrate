from . import _version
__version__ = _version.get_versions()['version']


from .energy_balance_model import multi_ebm, EnergyBalanceModel
from .fair import FAIR
