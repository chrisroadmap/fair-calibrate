"""
Module for ozone forcing
"""

import numpy as np
from ..constants.gases import BR_ATOMS, CL_ATOMS
from ..defaults.ozone import radiative_efficiency, br_cl_ratio, fractional_release
from ..defaults.forcing import tropospheric_adjustment
from ..defaults.gases import pre_industrial_concentration
from ..defaults.short_lived_forcers import pre_industrial_emissions
from ..defaults import gas_list, slcf_list

# important enough to stand alone
def calculate_eesc(
    concentration,
    pre_industrial_concentration,
    fractional_release=fractional_release,
    br_cl_ratio=br_cl_ratio,
):
    """Calculate equivalent effective stratospheric chlorine.

    TODO:
    Inputs
    ------
        concentration : dict of `np.ndarray` or float
            concentrations in timestep
        pre_industrial_concentration : dict of float
            pre-industrial concentrations
        fractional_release : dict of float
            fractional release describing the proportion of available ODS that
            actually contributes to ozone depletion.
        br_cl_ratio : float, default=45
            how much more effective bromine is as an ozone depletor than chlorine.

    Returns
    -------
        eesc_out : float
            equivalent effective stratospheric chlorine
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
        pre_industrial_emissions=pre_industrial_emissions,
        pre_industrial_concentration=pre_industrial_concentration,
        temperature=0,
        temperature_feedback=-0.037,
        radiative_efficiency=radiative_efficiency,
        timestep=1,
        br_cl_ratio=br_cl_ratio,
        fractional_release=fractional_release,
        tropospheric_adjustment=tropospheric_adjustment
    ):
    """Determines ozone effective radiative forcing.

    Calculates total ozone forcing from precursor emissions and
    concentrations based on AerChemMIP and CMIP6 Historical behaviour in
    Skeie et al. (2020) and Thornhill et al. (2021).

    In this hard-coded treatment, ozone forcing depends on concentrations of
    CH4, N2O, ozone-depleting halogens, and emissions of CO, NVMOC and NOx,
    but any combination of emissions and concentrations are allowed.

    Inputs
    ------
        emissions : dict of `np.ndarray` or float
            emissions in timestep
        concentration: dict of `np.ndarray` or float
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
        fractional_release : dict of float
            fractional release describing the proportion of available ODS that
            actually contributes to ozone depletion.

    Returns
    -------
        ozone forcing
    """

    radiative_forcing = {}
    effective_radiative_forcing = {}
    # for halogens
    eesc = calculate_eesc(
        concentration,
        pre_industrial_concentration,
        fractional_release=fractional_release,
        br_cl_ratio=br_cl_ratio,
    )
    radiative_forcing['Ozone|Emitted Gases|Montreal Gases'] = (
        eesc * radiative_efficiency['Montreal Gases']
    )

    def _linear_change(radeff, value, pi_value):
        return radeff * (value - pi_value)

    # to save time, we won't loop through all LLGHGs. But if in the future we
    # find that ozone is depleted by something else, this will need revisiting.
    species = {**emissions, **concentration}
    pre_industrial_species = {**pre_industrial_emissions, **pre_industrial_concentration}
    for specie in ['CH4', 'N2O', 'CO', 'VOC', 'NOx']:
        radiative_forcing['Ozone|Emitted Gases|{}'.format(specie)] = (
            _linear_change(
                radiative_efficiency[specie],
                species[specie],
                pre_industrial_species[specie]
            )
        )

    # needs to be in a constants module
    ods_species = ['CH4', 'N2O', 'CO', 'VOC', 'NOx', 'Montreal Gases']
    for igas, gas in enumerate(ods_species):
        if igas==0:
            radiative_forcing['Ozone|Emitted Gases'] = np.zeros_like(species[gas])
        radiative_forcing['Ozone|Emitted Gases'] = (
            radiative_forcing['Ozone|Emitted Gases'] +
            radiative_forcing['Ozone|Emitted Gases|{}'.format(gas)]
        )

    radiative_forcing['Ozone|Temperature Feedback'] = (
        temperature_feedback * temperature * radiative_forcing['Ozone|Emitted Gases']
    )
    print(temperature)
    print(radiative_forcing['Ozone|Emitted Gases'])
    print(radiative_forcing['Ozone|Temperature Feedback'])

    radiative_forcing['Ozone'] = (
        radiative_forcing['Ozone|Emitted Gases'] +
        radiative_forcing['Ozone|Temperature Feedback']
    )

    # I don't propose calculating ERF for every single component of o3 forcing
    effective_radiative_forcing['Ozone'] = (
        radiative_forcing['Ozone'] * tropospheric_adjustment['Ozone']
    )

    return radiative_forcing, effective_radiative_forcing
