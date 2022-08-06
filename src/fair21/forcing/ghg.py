import numpy as np

from ..constants import SPECIES_AXIS

def meinshausen2020(
    concentration,
    baseline_concentration,
    forcing_scaling,
    radiative_efficiency,
    co2_indices,
    ch4_indices,
    n2o_indices,
    minor_greenhouse_gas_indices,
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
    https://gmd.copernicus.org/articles/13/3571/2020/
    table 3

    Parameters
    ----------
    concentration : ndarray
        concentration of greenhouse gases. "CO2", "CH4" and "N2O" must be
        included in units of [ppm, ppb, ppb]. Other GHGs are units of ppt.
    baseline_concentration : ndarray
        pre-industrial concentration of the gases (see above).
    forcing_scaling : ndarray
        scaling of the calculated radiative forcing (e.g. for conversion to
        effective radiative forcing and forcing uncertainty).
    radiative_efficiency : ndarray
        radiative efficiency to use for linear-forcing gases, in W m-2 ppb-1
    co2_indices : np.ndarray of bool
        index along SPECIES_AXIS relating to CO2.
    ch4_indices : np.ndarray of bool
        index along SPECIES_AXIS relating to CH4.
    n2o_indices : np.ndarray of bool
        index along SPECIES AXIS relating to N2O.
    minor_greenhouse_gas_indices : np.ndarray of bool
        indices of other GHGs that are not CO2, CH4 or N2O.
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
    effective_radiative_forcing : np.ndarray
        effective radiative forcing (W/m2) from greenhouse gases

    References
    ----------
    [1] Meinshausen et al. 2020
    [2] Myhre et al. 1998
    """

    erf_out = np.ones_like(concentration) * np.nan

    # easier to deal with smaller arrays
    co2 = concentration[..., co2_indices]
    ch4 = concentration[..., ch4_indices]
    n2o = concentration[..., n2o_indices]
    co2_base = baseline_concentration[..., co2_indices]
    ch4_base = baseline_concentration[..., ch4_indices]
    n2o_base = baseline_concentration[..., n2o_indices]

    # CO2
    ca_max = co2_base - b1/(2*a1)
    where_central = np.asarray((co2_base < co2) & (co2 <= ca_max)).nonzero()
    where_low = np.asarray((co2 <= co2_base)).nonzero()
    where_high = np.asarray((co2 > ca_max)).nonzero()
    alpha_p = np.ones_like(co2) * np.nan
    alpha_p[where_central] = d1 + a1*(co2[where_central] - co2_base[where_central])**2 + b1*(co2[where_central] - co2_base[where_central])
    alpha_p[where_low] = d1
    alpha_p[where_high] = d1 - b1**2/(4*a1)
    alpha_n2o = c1*np.sqrt(n2o)
    erf_out[..., co2_indices] = (alpha_p + alpha_n2o) * np.log(co2/co2_base) * (forcing_scaling[..., co2_indices])

    # CH4
    erf_out[..., ch4_indices] = (
        (a3*np.sqrt(ch4) + b3*np.sqrt(n2o) + d3) *
        (np.sqrt(ch4) - np.sqrt(ch4_base))
    )  * (forcing_scaling[..., ch4_indices])

    # N2O
    erf_out[..., n2o_indices] = (
        (a2*np.sqrt(co2) + b2*np.sqrt(n2o) + c2*np.sqrt(ch4) + d2) *
        (np.sqrt(n2o) - np.sqrt(n2o_base))
    )  * (forcing_scaling[..., n2o_indices])

    # linear for other gases
    # TODO: move to a general linear function
    erf_out[..., minor_greenhouse_gas_indices] = (
        (concentration[..., minor_greenhouse_gas_indices] - baseline_concentration[..., minor_greenhouse_gas_indices])
        * radiative_efficiency[..., minor_greenhouse_gas_indices] * 0.001   # unit handling
    ) * (forcing_scaling[..., minor_greenhouse_gas_indices])

    return erf_out
