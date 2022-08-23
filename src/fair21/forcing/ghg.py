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

    Modified Etminan relationship from Meinshausen et al. (2020) [1]_
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
    .. [1] Meinshausen, M., Nicholls, Z.R.J., Lewis, J., Gidden, M.J.,
       Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N.,
       Canadell, J.G., Daniel, J.S., John, A., Krummel, P.B., Luderer, G.,
       Meinshausen, N., Montzka, S.A., Rayner, P.J., Reimann, S., Smith, S.J.,
       van den Berg, M., Velders, G.J.M., Vollmer, M.K., Wang, R.H.J. (2020).
       The shared socio-economic pathway (SSP) greenhouse gas concentrations
       and their extensions to 2500, Geoscientific Model Development, 13,
       3571–3605.
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


def myhre1998(
    concentration,
    baseline_concentration,
    forcing_scaling,
    radiative_efficiency,
    co2_indices,
    ch4_indices,
    n2o_indices,
    minor_greenhouse_gas_indices,
    alpha_co2=5.35,
    alpha_ch4=0.036,
    alpha_n2o=0.12,
    alpha_ch4_n2o=0.47,
    a1=2.01e-5,
    exp1=0.75,
    a2=5.32e-15,
    exp2=1.52
    ):
    """Greenhouse gas forcing from CO2, CH4 and N2O.

    Band overlaps are included between CH4 and N2O. This relationship comes from
    Myhre et al. (1998) [1]_, and was used up until the IPCC's Fifth Assessment
    Report.

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
    alpha_co2 : float, default=5.35
        factor relating logarithm of CO2 conentration to radiative forcing.
    alpha_ch4: float, default=5.35
        factor relating square root of CH4 conentration to radiative forcing.
    alpha_n2o : float, default=5.35
        factor relating square root of N2O conentration to radiative forcing.

    Returns
    -------
    effective_radiative_forcing : np.ndarray
        effective radiative forcing (W/m2) from greenhouse gases

    References
    ----------
    .. [1] Myhre, G., Highwood, E.J., Shine, K. Stordal, F. (1998). New
        estimates or radiative forcing due to well mixed greenhouse gases.
        Geophysical Research Letters, 25 (14), 2715-2718.
    """

    def ch4_n2o_overlap(ch4, n2o, alpha_ch4_n2o, a1, exp1, a2, exp2):
        return alpha_ch4_n2o * np.log(1 + a1 * (ch4*n2o)**exp1 + a2 * ch4 * (ch4*n2o)**exp2)

    erf_out = np.ones_like(concentration) * np.nan

    # easier to deal with smaller arrays
    co2 = concentration[..., co2_indices]
    ch4 = concentration[..., ch4_indices]
    n2o = concentration[..., n2o_indices]
    co2_base = baseline_concentration[..., co2_indices]
    ch4_base = baseline_concentration[..., ch4_indices]
    n2o_base = baseline_concentration[..., n2o_indices]

    # CO2
    erf_out[..., co2_indices] = alpha_co2 * np.log(co2/co2_base) * (forcing_scaling[..., co2_indices])

    # CH4
    erf_out[..., ch4_indices] = (
        alpha_ch4 *
        (np.sqrt(ch4) - np.sqrt(ch4_base)) -
        (ch4_n2o_overlap(ch4, n2o_base, alpha_ch4_n2o, a1, exp1, a2, exp2) - ch4_n2o_overlap(ch4_base, n2o_base, alpha_ch4_n2o, a1, exp1, a2, exp2))
    ) * forcing_scaling[..., ch4_indices]

    # N2O
    erf_out[..., n2o_indices] = (
        alpha_n2o *
        (np.sqrt(n2o) - np.sqrt(n2o_base)) -
        (ch4_n2o_overlap(ch4_base, n2o, alpha_ch4_n2o, a1, exp1, a2, exp2) - ch4_n2o_overlap(ch4_base, n2o_base, alpha_ch4_n2o, a1, exp1, a2, exp2))
    ) * forcing_scaling[..., n2o_indices]

    # linear for other gases
    # TODO: move to a general linear function
    erf_out[..., minor_greenhouse_gas_indices] = (
        (concentration[..., minor_greenhouse_gas_indices] - baseline_concentration[..., minor_greenhouse_gas_indices])
        * radiative_efficiency[..., minor_greenhouse_gas_indices] * 0.001   # unit handling
    ) * (forcing_scaling[..., minor_greenhouse_gas_indices])

    return erf_out
