import numpy as np

from ..defaults.gases import (
    pre_industrial_concentration,
    tropospheric_adjustment,
    radiative_efficiency
)

def meinshausen(
    concentration,
    pre_industrial_concentration=pre_industrial_concentration,
    tropospheric_adjustment=tropospheric_adjustment,
    a1 = -2.4785e-07,
    b1 = 0.00075906,
    c1 = -0.0021492,
    d1 = 5.2488,
    a2 = -0.00034197,
    b2 = 0.00025455,
    c2 = -0.00024357,
    d2 = 0.12173,
    a3 = -8.9603e-05,
    b3 = -0.00012462,
    d3 = 0.045194,
    ):
    """Greenhouse gas forcing from CO2, CH4 and N2O including band overlaps.

    Modified Etminan relationship from Meinshausen et al. (2020)
    https://gmd.copernicus.org/preprints/gmd-2019-222/gmd-2019-222.pdf
    table 3

    Inputs
    ------
        concentration : dict of float
            concentration of greenhouse gases. "CO2", "CH4" and "N2O" must be
            included in units of [ppm, ppb, ppb]. Other GHGs can be included
            but are not used.
        pre_industrial_concentration : dict of float
            pre-industrial concentration of the gases (see above).
        tropospheric_adjustment : dict of float
            conversion factor from radiative forcing to effective radiative forcing.
        a1 : float, default=-2.4785e-07
            fitting parameter (see Meinshausen et al. 2020)
        b1 : float, default=0.00075906
            fitting parameter (see Meinshausen et al. 2020)
        c1 : float, default=-0.0021492
            fitting parameter (see Meinshausen et al. 2020)
        d1 : float, default=5.2488
            fitting parameter (see Meinshausen et al. 2020)
        a2 : float, default=-0.00034197
            fitting parameter (see Meinshausen et al. 2020)
        b2 : float, default=0.00025455
            fitting parameter (see Meinshausen et al. 2020)
        c2 : float, default=-0.00024357
            fitting parameter (see Meinshausen et al. 2020)
        d2 : float, default=0.12173
            fitting parameter (see Meinshausen et al. 2020)
        a3 : float, default=-8.9603e-05
            fitting parameter (see Meinshausen et al. 2020)
        b3 : float, default=-0.00012462
            fitting parameter (see Meinshausen et al. 2020)
        d3 : float, default=0.045194
            fitting parameter (see Meinshausen et al. 2020)         

    Returns
    -------
        radiative_forcing : dict
            radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
        effective_radiative_forcing : dict
            effective radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
    """
    radiative_forcing = {}
    # CO2
    ca_max = pre_industrial_concentration["CO2"] - b1/(2*a1)
    if pre_industrial_concentration["CO2"] < concentration["CO2"] <= ca_max: # the most likely case
        alpha_p = d1 + a1*(concentration["CO2"] - pre_industrial_concentration["CO2"])**2 + b1*(concentration["CO2"] - pre_industrial_concentration["CO2"])
    elif concentration["CO2"] <= pre_industrial_concentration["CO2"]:
        alpha_p = d1
    else:
        alpha_p = d1 - b1**2/(4*a1)
    alpha_n2o = c1*np.sqrt(concentration["N2O"])
    radiative_forcing["CO2"] = (alpha_p + alpha_n2o) * np.log(concentration["CO2"]/pre_industrial_concentration["CO2"])

    # CH4
    radiative_forcing["CH4"] = (a3*np.sqrt(concentration["CH4"]) + b3*np.sqrt(concentration["N2O"]) + d3) * (np.sqrt(concentration["CH4"]) - np.sqrt(pre_industrial_concentration["CH4"]))

    # N2O
    radiative_forcing["N2O"] = (a2*np.sqrt(concentration["CO2"]) + b2*np.sqrt(concentration["N2O"]) + c2*np.sqrt(concentration["CH4"]) + d2) * (np.sqrt(concentration["N2O"]) - np.sqrt(pre_industrial_concentration["N2O"]))

    # effective radiative forcing
    effective_radiative_forcing = {}
    for gas in ["CO2", "CH4", "N2O"]:
        effective_radiative_forcing[gas] = radiative_forcing[gas] * tropospheric_adjustment[gas]

    return radiative_forcing, effective_radiative_forcing


def etminan(
    concentration,
    pre_industrial_concentration=pre_industrial_concentration,
    tropospheric_adjustment=tropospheric_adjustment
):
    """Greenhouse gas forcing from CO2, CH4 and N2O including band overlaps.
    
    This function uses the updated formulas of Etminan et al. (2016),
    including the overlaps between CO2, methane and nitrous oxide.
    Reference: Etminan et al, 2016, JGR, doi: 10.1002/2016GL071930
    
    Inputs
    ------
        concentration : dict of float
            concentration of greenhouse gases. "CO2", "CH4" and "N2O" must be
            included in units of [ppm, ppb, ppb]. Other GHGs can be included
            but are not used.
        pre_industrial_concentration : dict of float
            pre-industrial concentration of the gases (see above).
        tropospheric_adjustment : dict of float
            conversion factor from radiative forcing to effective radiative forcing.
            
    Returns
    -------
        radiative_forcing : dict
            radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
        effective_radiative_forcing : dict
            effective radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
    """
    
    c_bar = 0.5 * (concentration["CO2"] + pre_industrial_concentration["CO2"])
    m_bar = 0.5 * (concentration["CH4"] + pre_industrial_concentration["CH4"])
    n_bar = 0.5 * (concentration["N2O"] + pre_industrial_concentration["N2O"])

    radiative_forcing = {}
    radiative_forcing["CO2"] = (
        (-2.4e-7*(concentration["CO2"] - pre_industrial_concentration["CO2"])**2 + 
        7.2e-4*np.fabs(concentration["CO2"]-pre_industrial_concentration["CO2"]) - 
        2.1e-4 * n_bar + 5.36) * np.log(concentration["CO2"]/pre_industrial_concentration["CO2"])
    )
    radiative_forcing["CH4"] = (
        (-1.3e-6*m_bar - 8.2e-6*n_bar + 0.043) * 
        (np.sqrt(concentration["CH4"]) - 
        np.sqrt(pre_industrial_concentration["CH4"]))
    )
    radiative_forcing["N2O"] = (-8.0e-6*c_bar + 4.2e-6*n_bar - 4.9e-6*m_bar + 0.117) * (np.sqrt(concentration["N2O"]) - np.sqrt(pre_industrial_concentration["N2O"]))

    effective_radiative_forcing = {}
    for gas in ["CO2", "CH4", "N2O"]:
        effective_radiative_forcing[gas] = radiative_forcing[gas] * tropospheric_adjustment[gas]

    return radiative_forcing, effective_radiative_forcing


# what are we doing about ERF here?
def myhre(
    concentration,
    pre_industrial_concentration=pre_industrial_concentration,
    radiative_forcing_2co2=3.71
):
    """Calculate the radiative forcing from CO2, CH4 and N2O.
    
    This uses the Myhre et al. (1998) relationships including the band
    overlaps between CH4 and N2O. It is also used in AR5.
    Reference: Myhre et al, 1998, JGR, doi: 10.1029/98GL01908
    
    Inputs
    ------
        concentration : dict of float
            concentration of greenhouse gases. "CO2", "CH4" and "N2O" must be
            included in units of [ppm, ppb, ppb]. Other GHGs can be included
            but are not used.
        pre_industrial_concentration : dict of float
            pre-industrial concentration of the gases (see above).
        tropospheric_adjustment : dict of float
            conversion factor from radiative forcing to effective radiative forcing.
    Returns:
        3-element array of radiative forcing: [F_CO2, F_CH4, F_N2O]
    """

    radiative_forcing = {}

    radiative_forcing["CO2"] = co2_log(concentration["CO2"], pre_industrial_concentration["CO2"], radiative_forcing_2co2)
    
    radiative_forcing["CH4"] = 0.036 * (np.sqrt(concentration["CH4"]) - np.sqrt(pre_industrial_concentration["CH4"])) - (
      _ch4_n2o_overlap(concentration["CH4"], concentration["N2O"]) - _ch4_n2o_overlap(pre_industrial_concentration["CH4"], pre_industrial_concentration["N2O"]))
    radiative_forcing["N2O"] = 0.12 * (np.sqrt(concentration["N2O"]) - np.sqrt(pre_industrial_concentration["N2O"])) - (
      _ch4_n2o_overlap(pre_industrial_concentration["CH4"],concentration["N2O"]) - _ch4_n2o_overlap(pre_industrial_concentration["CH4"],pre_industrial_concentration["N2O"])) 

    return radiative_forcing


def _ch4_n2o_overlap(M, N):
    return 0.47 * np.log(1 + 2.01e-5*(M*N)**(0.75) + 5.31e-15*M*(M*N)**(1.52))


# should be able to take dict or float
# what are we doing about ERF here?
def co2_log(
    concentration,
    pre_industrial_concentration=pre_industrial_concentration["CO2"],
    radiative_forcing_2co2=3.71
):
    """Calculates radiative forcing from CO2 using old logarithmic formula.
    
    Inputs
    ------
    concentration : float
        concentration of CO2, ppm
    pre_industrial_concentration : float
        pre-industrial concentration of CO2, ppm
    radiative_forcing_2co2 : float
        radiative forcing (W/m2) from a doubling of CO2.
    
    Returns
    -------
    radiative_forcing : float
        radiative forcing (W/m2) of CO2.
    """
    return radiative_forcing_2co2/np.log(2) * np.log(concentration/pre_industrial_concentration)


def linear(
    concentration,
    pre_industrial_concentration=pre_industrial_concentration,
    radiative_efficiency=radiative_efficiency,
    tropospheric_adjustment=tropospheric_adjustment
):
    """Greenhouse gas forcing from linear change in concentrations.
    
    This function is useful for GHGs that are not CO2, CH4 and N2O, in which
    there is no good reason to assume that the change in radiative forcing is
    not linear with concentrations. This treatment has been used since the 
    1990s and through to AR5 and AR6.
    
    Inputs
    ------
        concentration : dict of float
            concentrations of greenhouse gases.
        pre_industrial_concentration : dict of float
            pre-industrial concentration of the gases (see above).
        radiative_efficiency : dict of float
            radiative efficiency (W/m2/ppb) for each gas given in input.
        tropospheric_adjustment : dict of float
            conversion factor from radiative forcing to effective radiative forcing.
            
    Returns
    -------
        radiative_forcing : dict
            radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
        effective_radiative_forcing : dict
            effective radiative forcing (W/m2) of "CO2", "CH4" and "N2O".
    """
    
    radiative_forcing = {}
    effective_radiative_forcing = {}
    for gas in concentration.keys():
        # we should actually remove this condition to allow for metric calculations around a present day baseline
        if gas in ["CO2", "CH4", "N2O"]:
            continue
        radiative_forcing[gas] = (concentration[gas] - pre_industrial_concentration[gas]) * radiative_efficiency[gas] * 0.001
        effective_radiative_forcing[gas] = radiative_forcing[gas] * tropospheric_adjustment[gas]
    # """
    # Calculate radiative forcing from minor gas species.
    # Inputs:
        # C: concentration of minor GHGs (in order of MAGICC RCP concentration
            # spreadsheets)
        # Cpi: Pre-industrial concentration of GHGs
    # Returns:
        # 28 element array of minor GHG forcings
    # """

    # return (C - Cpi) * radeff.aslist[3:] * 0.001
