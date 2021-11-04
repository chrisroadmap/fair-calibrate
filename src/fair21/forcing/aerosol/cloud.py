import numpy as np

from ...defaults.aerosol.cloud import beta, shape

# TODO: a generalised log relationship and inclusion of temperature
# feedback. This relationship is untested as yet on ESMs.

def smith2021(
    emissions,
    pre_industrial_emissions,
    beta=beta["AR6"],
    shape_sulfur=shape["AR6"]["Sulfur"],
    shape_bcoc=shape["AR6"]['BC+OC']
):
    """ERF from aerosol-cloud interactions

    This uses the relationship to calculate ERFaci described in Smith et al.
    (2021).

    Inputs
    ------
    emissions : dict of `np.ndarray` or float
        emissions of short_lived_forcers
    pre_industrial_emissions : dict
        pre-industrial values of emissions
    beta : float
        scaling factor converting to ERFaci.
    shape_sulfur : float
        shape factor describing the "logarithmicity" of the behaviour of
        ERFaci with sulfur emissions.
    shape_bcoc : float
        shape factor describing the "logarithmicity" of the behaviour of
        ERFaci with the sum of BC and OC emissions.

    Returns
    -------
    erf_aci :
        Effective radiative forcing due to aerosol-cloud interactions.
    """

    radiative_effect = -beta * np.log(
        1 + emissions["Sulfur"]/shape_sulfur +
        (emissions["BC"] + emissions["OC"])/shape_bcoc
    )
    pre_industrial_radiative_effect = -beta * np.log(
        1 + pre_industrial_emissions["Sulfur"]/shape_sulfur +
        (pre_industrial_emissions["BC"] + pre_industrial_emissions["OC"])/shape_bcoc
    )
    return radiative_effect - pre_industrial_radiative_effect


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
