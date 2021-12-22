import numpy as np

from ...defaults.aerosol import beta, shape

# TODO: a generalised log relationship and inclusion of temperature
# feedback. This relationship is untested as yet on ESMs.

def smith2021(
    emissions,
    pre_industrial_emissions,
    tropospheric_adjustment,
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
    tropospheric_adjustment : ndarray
        conversion factor from radiative forcing to effective radiative forcing.
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

    erf_out = (radiative_effect - pre_industrial_radiative_effect) * (1 + tropospheric_adjustment)
    return erf_out


### NEEDS UPDATING
def stevens2015(
    emissions,
    pre_industrial_emissions,
    beta=beta["Stevens2015"],
    shape_sulfur=shape["Stevens2015"]
):
    """ERF from aerosol-cloud interactions

    This uses the relationship to calculate ERFaci described in Stevens (2015).

    Inputs
    ------
    emissions : dict of `np.ndarray` or float
        emissions of short_lived_forcers
    pre_industrial_emissions : dict
        pre-industrial values of emissions
    beta : float
        scaling factor converting to ERFaci.
    shape_sulfur : float
        natural emissions of SO2 (parallels to `shape` parameter of Smith et al.
        2021).

    Returns
    -------
    erf_aci :
        Effective radiative forcing due to aerosol-cloud interactions.
    """

    radiative_effect = -beta * np.log(
        1 + emissions["Sulfur"]/shape_sulfur
    )
    pre_industrial_radiative_effect = -beta * np.log(
        1 + pre_industrial_emissions["Sulfur"]/shape_sulfur
    )
    return radiative_effect - pre_industrial_radiative_effect
