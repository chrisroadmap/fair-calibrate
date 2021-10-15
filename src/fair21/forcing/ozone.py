"""
Module for ozone forcing
"""

import numpy as np
#from ..constants import cl_atoms, br_atoms, fracrel
from ..defaults.ozone import radiative_efficiency, br_cl_ratio, fractional_release
from ..constants.gases import CL_ATOMS, BR_ATOMS

# important enough to stand alone
def calculate_eesc(
    concentration,
    pre_industrial_concentration,
    fractional_release=fractional_release
    br_cl_ratio=br_cl_ratio,
):
    """Calculate equivalent effective stratospheric chlorine.
    
    TODO:
    Inputs
    ------
    
    Returns
    -------
    """
    
    for igas, gas in enumerate(concentration.keys()):
        if igas==0:
            eesc_out = np.zeros_like(concentration[gas])
        eesc_out = eesc_out + (
            CL_ATOMS[gas] * (concentration[gas] - pre_industrial_concentration[gas]) * fractional_release[gas] / fractional_release["CFC-11"] +
            br_cl_ratio * BR_ATOMS[gas] * (concentration[gas] - pre_industrial_concentration[gas]) * fractional_release[gas] / fractional_release["CFC-11"]
        ) * fractional_release["CFC-11"]
    return eesc_out


def thornhill_skeie(
        emissions,
        concentration,
        temperature=0,
        temperature_feedback=-0.037,
        radiative_efficiency=radiative_efficiency,
        timestep=1,
        br_cl_ratio=br_cl_ratio,
        fractional_release=fractional_release
#        emissions_pi=np.zeros(40),
#        concentrations_pi=np.zeros(31),
    ):
    """Determines ozone effective radiative forcing.
    
    Calculates total ozone forcing from precursor emissions and
    concentrations based on AerChemMIP and CMIP6 Historical behaviour
    Skeie et al. (2020) and Thornhill et al. (2021).

    By default, ozone forcing depends on concentrations of
    CH4, N2O, ozone-depleting halogens, and emissions of CO, NVMOC and NOx,
    but any combination of emissions and concentrations are allowed.
    
    Inputs
    ------
        emissions : dict of `np.ndarray` or float
            emissions in timestep
        concentrations: dict of `np.ndarray` or float
            concentrations in timestep
        temperature : float
            global mean surface temperature anomaly in the previous timestep.
            A future TODO could be to iterate this.
        temperature_feedback : float
            temperature feedback on ozone forcing (W/m2/K)
        radiative_efficiency : dict of float
            radiative efficiencies of ozone forcing to different emitted species
            or atmospheric concentrations. Units should be (W/m/[unit]) where
            [unit] is the emissions or concentration unit.
        br_cl_ratio : float, default=45
            how much more effective bromine is as an ozone depletor than chlorine.

    Returns
    -------
        ozone forcing
    """

    # calculate EESC for halogens
    cl = np.array(cl_atoms.aslist)
    br = np.array(br_atoms.aslist)
    fc = np.array(fracrel.aslist)

    eesc = calculate_eesc(
        concentration,
        pre_industrial_concentration,
        fractional_release=fractional_release
        br_cl_ratio=br_cl_ratio,
    )

    c_ch4, c_n2o = concentrations[:, [1, 2]].T
#    delta_c_ods = eesc(concentrations[:,15:].T, concentrations_pi[None, 15:])
    c_ods = concentrations[:,15:]
    e_co, e_nmvoc, e_nox = emissions[:,[6, 7, 8]].T
    c_ch4_pi, c_n2o_pi = concentrations_pi[[1, 2]]
    c_ods_pi = concentrations_pi[15:]
    e_co_pi, e_nmvoc_pi, e_nox_pi = emissions_pi[[6, 7, 8]]


    forcing = np.zeros(nt)
    if np.isscalar(temperature):
        temperature = np.ones(nt) * temperature

    for i in range(nt):
        f_ch4   = beta[0] * (c_ch4[i] - c_ch4_pi)
        f_n2o   = beta[1] * (c_n2o[i] - c_n2o_pi)
        f_ods   = beta[2] * eesc(c_ods[i], c_ods_pi)
        f_co    = beta[3] * (e_co[i] - e_co_pi)
        f_nmvoc = beta[4] * (e_nmvoc[i] - e_nmvoc_pi)
        f_nox   = beta[5] * (e_nox[i] - e_nox_pi)
        forcing[i] = f_ch4 + f_n2o + f_ods + f_co + f_nmvoc + f_nox + feedback * temperature[i]

    return radiative_forcing