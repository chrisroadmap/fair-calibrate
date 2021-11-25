"""
Module for the forward (emissions to concentration) model.
"""

import numpy as np

from ..constants.general import GAS_BOX_AXIS

def step_concentration(
    emissions,
    gas_boxes_old,
    airborne_emissions_old,
    burden_per_emission,
    lifetime,
    alpha_lifetime,
    partition_fraction,
    pre_industrial_concentration,
    timestep=1,
    natural_emissions_adjustment=0,
):
    """
    Calculates concentrations from emissions of any greenhouse gas.

    Parameters
    ----------
    emissions : ndarray
        emissions in timestep.
    gas_boxes_old : ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the previous timestep.
    airborne_emissions_old : ndarray
        The total airborne emissions at the beginning of the timestep. This is
        the concentrations above the pre-industrial control. It is also the sum
        of gas_boxes_old if this is an array.
    burden_per_emission : ndarray
        how much atmospheric concentrations grow (e.g. in ppm) per unit (e.g.
        GtCO2) emission.
    lifetime : ndarray
        atmospheric burden lifetime of greenhouse gas (yr). For multiple
        lifetimes gases, it is the lifetime of each box.
    alpha_lifetime : ndarray
        scaling factor for `lifetime`. Necessary where there is a state-
        dependent feedback.
    partition_fraction : ndarray
        the partition fraction of emissions into each gas box. If array, the
        entries should be individually non-negative and sum to one.
    pre_industrial_concentration : ndarray
        pre-industrial concentration of gas(es) in question.
    timestep : float, default=1
        emissions timestep in years.
    natural_emissions_adjustment : ndarray or float, default=0
        Amount to adjust emissions by for natural emissions given in the total
        in emissions files.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
    on the same time intervals: the airborne_emissions values are on time
    boundaries and these are averaged before being returned.

    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    concentration_out : ndarray
        greenhouse gas concentrations at the centre of the timestep.
    gas_boxes_new : ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the timestep.
    airborne_emissions_new : ndarray
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """

    decay_rate = timestep/(alpha_lifetime * lifetime)
    decay_factor = np.exp(-decay_rate)

    gas_boxes_new = (
        partition_fraction *
        (emissions-natural_emissions_adjustment) *
        1 / decay_rate *
        (1 - decay_factor) * timestep + gas_boxes_old * decay_factor
    )
    airborne_emissions_new = np.sum(gas_boxes_new, axis=GAS_BOX_AXIS, keepdims=True)
    concentration_out = (
        pre_industrial_concentration +
        burden_per_emission * (
            airborne_emissions_new + airborne_emissions_old
        ) / 2
    )

    return concentration_out, gas_boxes_new, airborne_emissions_new
