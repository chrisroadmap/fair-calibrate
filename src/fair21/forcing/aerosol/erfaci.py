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
    so2_index,
    bc_index,
    oc_index,
    aci_method,
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
    so2_index : int or None
        array index along SPECIES_AXIS corresponding to SO2 emissions.
    bc_index : int or None
        array index along SPECIES_AXIS corresponding to BC emissions.
    oc_index : int or None
        array index along SPECIES_AXIS corresponding to OC emissions.
    aci_method : ACIMethod
        Method used to calculate aerosol forcing.

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

    The Stevens (2015) [1]_ relationship is the second term of their eq. (1):

    :math: F_{aci} = -\beta \log \left \frac{E_{SO2}}{s_{SO2}} + 1 \right

    The Smith et al. (2018) [2]_ relationship as formulated more explicitly in
    Smith et al. (2021) [3]_ is

    :math: F_{aci} = -\beta \log \left \frac{E_{SO2}}{s_{SO2}} + \frac{E_{BC+OC}}{s_{BC+OC}} + 1 \right

    (note there is a typo in Smith et al. 2021).

    References
    ----------
    .. [1] Stevens, B. (2015). Rethinking the Lower Bound on Aerosol Radiative
        Forcing, Journal of Climate, 28(12), 4794-4819.

    .. [2] Smith, C. J., Forster, P. M.,  Allen, M., Leach, N., Millar, R. J.,
        Passerello, G. A., and Regayre, L. A. (2018). FAIR v1.3: a simple
        emissions-based impulse response and carbon cycle model, Geosci. Model
        Dev., 11, 2273–2297

    .. [3] Smith, C. J., Harris, G. R., Palmer, M. D., Bellouin, N., Collins,
        W., Myhre, G., Schulz, M., Golaz, J.-C., Ringer, M., Storelvmo, T.,
        Forster, P. M. (2021). Energy budget constraints on the time history of
        aerosol forcing and climate sensitivity. Journal of Geophysical
        Research: Atmospheres, 126, e2020JD033622.
    """

    so2 = emissions[:, :, :, [so2_index], ...]
    so2_base = pre_industrial_emissions[:, :, :, [so2_index], ...]

    if aci_method==ACIMethod.SMITH2018:
        bc = emissions[:, :, :, [bc_index], ...]
        bc_base = pre_industrial_emissions[:, :, :, [bc_index], ...]
        oc = emissions[:, :, :, [oc_index], ...]
        oc_base = pre_industrial_emissions[:, :, :, [oc_index], ...]

    else:
        bc = bc_base = oc = oc_base = 0
        shape_bcoc = 100  # anything to avoid divide by zero

    # TODO: raise an error if sulfur, BC and OC are not all there
    radiative_effect = -scale * np.log(
        1 + so2/shape_sulfur +
        (bc + oc)/shape_bcoc
    )
    pre_industrial_radiative_effect = -scale * np.log(
        1 + so2_base/shape_sulfur +
        (bc_base + oc_base)/shape_bcoc
    )

    erf_out = (radiative_effect - pre_industrial_radiative_effect) * forcing_scaling
    return erf_out
