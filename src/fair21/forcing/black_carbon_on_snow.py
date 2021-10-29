"""
Module for black carbon on snow forcing
"""

from ..defaults.black_carbon_on_snow import radiative_efficiency
from ..defaults.short_lived_forcers import pre_industrial_emissions

def linear(
    emissions,
    pre_industrial_emissions=pre_industrial_emissions,
    radiative_efficiency=radiative_efficiency["AR6"]
):
    """
    Calculate effective radiative forcing from black carbon on snow.

    This makes the assumption that the land and sea ice aereal extent does not
    significantly change with temperature. One would assume that the available
    area for black carbon to settle on would decrease with warming, but given
    that BC on snow is a small forcing and limited research has been undertaken
    in this area, no temperature dependence is included.

    Inputs
    ------
    emissions : dict or `np.ndarray` or float
        input emissions
    pre_industrial_emissions : dict or float
        pre-industrial emissions
    radiative_efficiency : dict or float
        radiative efficiency of each species.

    Returns
    -------
    erf_bc_snow : dict
        ERF from black carbon on snow.
    """
    emissions_bc = emissions
    pre_industrial_emissions_bc = pre_industrial_emissions
    if isinstance(emissions, dict):
        emissions_bc = emissions['BC']
    if isinstance(pre_industrial_emissions, dict):
        pre_industrial_emissions_bc = pre_industrial_emissions['BC']
    erf_bc_snow = (
        radiative_efficiency *
        (emissions_bc - pre_industrial_emissions_bc)
    )
    return erf_bc_snow
