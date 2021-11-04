import numpy as np

from ..defaults.landuse import forcing_from_cumulative_co2_afolu

def from_cumulative_co2(
    cumulative_emissions_co2,
    pre_industrial_cumulative_emissions_co2=0,
    forcing_from_cumulative_co2_afolu=forcing_from_cumulative_co2_afolu["AR6"]
):
    """Estimate land-use effective radiative forcing.

    This is a simplified expression that is a linear function of cumulative
    AFOLU CO2 emissions, which is found to be the case in MAGICC6.

    Parameters
    ----------
    cumulative_emissions_co2 : ndarray or float
        Cumulative emissions of CO2 in GtCO2.
    pre_industrial_cumulative_emissions_co2 : float
        Pre-industrial baseline value to use (GtCO2), which will be subtracted
        from the cumulative emissions.
    forcing_from_cumulative_co2_afolu : float, optional
        conversion factor to translate to effective radiative forcing
        units (W m-2 GtCO2-1)

    Returns
    -------
    ndarray or float :
        effective radiative forcing from land use.
    """

    return (cumulative_emissions_co2 - pre_industrial_cumulative_emissions_co2) * forcing_from_cumulative_co2_afolu
