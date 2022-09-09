"""Module for forcing from aerosol-cloud interactions."""

import numpy as np

from ...constants import SPECIES_AXIS


def leach2021aci(
    emissions,
    baseline_emissions,
    forcing_scaling,
    scale_sulfur,
    shape_sulfur,
    scale_bcoc,
    sulfur_index,
    bc_index,
    oc_index,
):
    """Calculate effective radiative forcing from aerosol-cloud interactions.

    This uses the relationship to calculate ERFaci described in Leach et al.
    (2021) [1]_ and used in FaIRv2.0.

    Inputs
    ------
    emissions : ndarray
        input emissions
    baseline_emissions : ndarray
        pre-industrial emissions
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    scale_sulfur : ndarray
        scaling factor to apply to the logarithm of sulfur
    shape_sulfur : ndarray
        shape factor for sulfur emissions
    scale_bcoc : ndarray
        scale factor to apply to linear BC+OC emissions
    sulfur_index : int or None
        array index along SPECIES_AXIS corresponding to SO2 emissions.
    bc_index : int or None
        array index along SPECIES_AXIS corresponding to BC emissions.
    oc_index : int or None
        array index along SPECIES_AXIS corresponding to OC emissions.

    Returns
    -------
    erf_out : ndarray
        effective radiative forcing (W/m2) from aerosol-cloud interactions

    References
    ----------
    .. [1] Leach, N.J., Jenkins, S., Nicholls, Z., Smith, C.J., Lynch, J.,
        Cain, M., Walsh, T., Wu, B., Tsutsui, J., Allen, M.R. (2021). FaIRv2.0.0:
        a generalized impulse response model for climate uncertainty and future
        scenario exploration, Geoscientific Model Development, 14, 3007–3036.
    """
    # this allows us to run single forced
    array_shape = emissions.shape
    n_timesteps, n_scenarios, n_configs, n_species = array_shape
    _erf = np.ones((array_shape)) * np.nan

    sulfur = emissions[..., sulfur_index]
    sulfur_base = baseline_emissions[..., sulfur_index]
    bc = emissions[..., bc_index]
    bc_base = baseline_emissions[..., bc_index]
    oc = emissions[..., oc_index]
    oc_base = baseline_emissions[..., oc_index]

    # sulfur contribution
    _erf[..., sulfur_index] = (-scale_sulfur * np.log(1 + sulfur / shape_sulfur)) - (
        -scale_sulfur * np.log(1 + sulfur_base / shape_sulfur)
    )

    # bc contribution
    _erf[..., bc_index] = scale_bcoc * (bc - bc_base)

    # oc contribution
    _erf[..., oc_index] = scale_bcoc * (oc - oc_base)

    erf_out = np.nansum(_erf, axis=SPECIES_AXIS, keepdims=True) * forcing_scaling
    return erf_out


def smith2022(
    emissions,
    baseline_emissions,
    forcing_scaling,
    scale,
    shape,
):
    r"""Calculate effective radiative forcing from aerosol-cloud interactions.

    This uses the relationship to calculate ERFaci as follows

    F = \sum_{i} \beta_i \log \left( 1 + \frac{E_i}{n_i} \right)

    where
        $A_i$ is the emissions of a specie
        $\beta_i$ is the scale factor
        $n_i$ is a shape factor that describes how logarithmic the relationship is.

    Inputs
    ------
    emissions : ndarray
        input emissions
    baseline_emissions : ndarray
        baseline emissions
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    scale : ndarray
        per-species scaling factor to apply to the logarithm
    shape : ndarray
        per-species shape factor for the logarithm

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from aerosol-cloud interactions
    """
    radiative_effect = np.nansum(
        scale * np.log(1 + emissions / shape), axis=SPECIES_AXIS
    )
    baseline_radiative_effect = np.nansum(
        scale * np.log(1 + baseline_emissions / shape), axis=SPECIES_AXIS
    )

    erf_out = (radiative_effect - baseline_radiative_effect) * forcing_scaling
    return erf_out


def stevens2015(
    emissions,
    baseline_emissions,
    forcing_scaling,
    scale,
    shape_sulfur,
    sulfur_index,
):
    r"""Calculate effective radiative forcing from aerosol-cloud interactions.

    This uses the relationship to calculate ERFaci described in Stevens (2015) [1]_.

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
    sulfur_index : int or None
        array index along SPECIES_AXIS corresponding to SO2 emissions.

    Returns
    -------
    effective_radiative_forcing : ndarray
        effective radiative forcing (W/m2) from aerosol-cloud interactions

    Notes
    -----
    The Stevens (2015) [1]_ relationship is the second term of their eq. (1):

    :math: F_{aci} = -\beta \log \left \frac{E_{SO2}}{s_{SO2}} + 1 \right

    References
    ----------
    .. [1] Stevens, B. (2015). Rethinking the Lower Bound on Aerosol Radiative
        Forcing, Journal of Climate, 28(12), 4794-4819.
    """
    sulfur = emissions[..., sulfur_index]
    sulfur_base = baseline_emissions[..., sulfur_index]

    radiative_effect = -scale * np.log(1 + sulfur / shape_sulfur)
    baseline_radiative_effect = -scale * np.log(1 + sulfur_base / shape_sulfur)

    erf_out = (radiative_effect - baseline_radiative_effect) * forcing_scaling
    return erf_out
