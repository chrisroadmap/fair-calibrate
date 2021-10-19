"""
Module for forcing from aviation contrails and contrail-induced cirrus.
"""

import numpy as np
from ..defaults.contrail import radiative_efficiency, pre_industrial_emissions

def linear_from_aviation_nox(
    emissions,
    pre_industrial_emissions=pre_industrial_emissions,
    radiative_efficiency=radiative_efficiency["AR6"]
):
    """Calculates contrail radiative forcing from emissions of aviation NOx.

    Inputs
    ------
    emissions : dict or `np.ndarray` or float
        Emissions of aviation NOx. If a dict of emissions is specified, the
        key "NOx|MAGICC Fossil and Industrial|Aircraft" will
        be searched for. This has nothing to do with MAGICC; it's an RCMIP
        convention.
    pre_industrial_emissions : dict or float
        Pre-industrial emissions of aviation NOx (see above).
    radiative_efficiency : float
        radiative efficiency (W m-2 (Mt NO2 yr-1)-1)

    Returns
    -------
    erf_contrail :
        Contrail forcing
    """
    emissions_aviation_nox = emissions
    pre_industrial_emissions_aviation_nox = pre_industrial_emissions
    if isinstance(emissions, dict):
        emissions_aviation_nox = emissions['NOx|MAGICC Fossil and Industrial|Aircraft']
    if isinstance(pre_industrial_emissions, dict):
        pre_industrial_emissions_aviation_nox = pre_industrial_emissions['NOx|MAGICC Fossil and Industrial|Aircraft']
    erf_contrail = (
        radiative_efficiency *
        (emissions_aviation_nox - pre_industrial_emissions_aviation_nox)
    )
    return erf_contrail
