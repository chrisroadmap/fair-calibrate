import numpy as np

from ...exceptions import IncompatibleConfigError
from ...structure.top_level import ACIMethod

def _check_aci_params(aci_params, aci_method):
    required_params = {
        ACIMethod.SMITH2018: ['scale', 'Sulfur', 'BC+OC'],
        ACIMethod.STEVENS2015: ['scale', 'Sulfur']
    }
    for param in required_params[aci_method]:
        if param not in aci_params:
            raise IncompatibleConfigError(
                f"For aerosol-cloud interactions using the {aci_method}, "
                f"the aci_params in the construction of Config must include "
                f"{required_params[aci_method]}."
            )

def calculate_erfaci_forcing(
    emissions,
    pre_industrial_emissions,
    forcing_scaling,
    scale,
    shape_sulfur,
    shape_bcoc,
    aerosol_index_mapping,
):
    """Calculate effective radiative forcing from aerosol-cloud interactions.

    This uses the relationship to calculate ERFaci described in Smith et al.
    (2021).

    Inputs
    ------
    emissions : ndarray
        input emissions
    pre_industrial_emissions : ndarray
        pre-industrial emissions
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    scale : ndarray
        scaling factor to apply to the logarithm
    shape_sulfur : ndarray
        scale factor for sulfur emissions
    shape_bcoc : ndarray
        scale factor for BC+OC emissions
    radiative_efficiency : ndarray
        radiative efficiency (W m-2 (emission_unit yr-1)-1) of each species.
    aerosol_index_mapping : dict
        provides a mapping of which aerosol species corresponds to which array
        index along the SPECIES_AXIS.

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from aerosol-cloud interactions

    Notes
    -----
    Where array input is taken, the arrays always have the dimensions of
    (time, scenario, config, species, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.
    """

    so2 = emissions[:, :, :, [aerosol_index_mapping["Sulfur"]], ...]
    so2_pi = pre_industrial_emissions[:, :, :, [aerosol_index_mapping["Sulfur"]], ...]
    bc = emissions[:, :, :, [aerosol_index_mapping["BC"]], ...]
    bc_pi = pre_industrial_emissions[:, :, :, [aerosol_index_mapping["BC"]], ...]
    oc = emissions[:, :, :, [aerosol_index_mapping["OC"]], ...]
    oc_pi = pre_industrial_emissions[:, :, :, [aerosol_index_mapping["OC"]], ...]
    aci_index = aerosol_index_mapping["Aerosol-Cloud Interactions"]


    # TODO: raise an error if sulfur, BC and OC are not all there
    radiative_effect = -scale * np.log(
        1 + so2/shape_sulfur +
        (bc + oc)/shape_bcoc
    )
    pre_industrial_radiative_effect = -scale * np.log(
        1 + so2_pi/shape_sulfur +
        (bc_pi + oc_pi)/shape_bcoc
    )

    erf_out = (radiative_effect - pre_industrial_radiative_effect) * forcing_scaling
    return erf_out
