"""
Tools for the FaIR gas cycle.
"""

import numpy as np

# This is now somewhat redundant
def lifetime_to_iirf_0(lifetime, iirf_horizon=100):
    """Converts greenhouse gas lifetime to time-integrated airborne fraction.

    iirf_0 is the 100-year time-integrated airborne fraction to a pulse
    emission. We know that the gas's atmospheric airborne fraction $a(t)$ for a
    gas with lifetime $\tau$ after time $t$ is therefore

    $a(t) = \exp(-t/tau)$

    and integrating this for 100 years after a pulse emissions gives us:

    $r_0(t) = \int_0^{100} \exp(-t/\tau) dt = \tau (1 - \exp (-100 / \tau))$.

    100 years is the default time horizon in FaIR but this can be set to any
    value.

    Parameters
    ----------
    lifetime : float
        greenhouse gas atmospheric lifetime (yr)
    iirf_horizon : float, optional
        time horizon for time-integrated airborne fraction (yr).

    Returns
    -------
    iirf_0 : float
        time-integrated airborne fraction
    """

    return (lifetime * (1 - np.exp(-iirf_horizon / lifetime)))
