"""
Module for inverting concentrations.
"""

import numpy as np

from ..constants import GAS_BOX_AXIS

def unstep_concentration(
    concentration,
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
    Calculates emissions from concentrations of any greenhouse gas.

    Parameters
    ----------
    concentration : ndarray
        greenhouse gas concentration at the centre of the timestep.
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
        pre-industrial concentration gas in question.
    timestep : float, default=1
        emissions timestep in years.
    natural_emissions_adjustment : ndarray or float, default=0
        Amount to adjust emissions by for natural emissions given in the total
        in emissions files.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
    on the same time intervals: the airborne_emissions values are on time
    boundaries. Therefore it is not actually possible to provide the exact
    emissions that would reproduce the concentrations without using a slower
    root-finding mechanism (that was present in v1.4) and will always be half
    a time step out.

    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    emissions_out : ndarray
        emissions in timestep.
    gas_boxes_new : ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the timestep.
    airborne_emissions_new : ndarray
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """

    # comments are keeping track of units
    decay_rate = timestep/(alpha_lifetime * lifetime)   # [1]
    decay_factor = np.exp(-decay_rate)  # [1]

    # [GtCO2] = [ppm] - [ppm] / [ppm/GtCO2]
    airborne_emissions_new = (concentration-pre_industrial_concentration)/burden_per_emission

    # [GtCO2/yr] = [GtCO2] - [GtCO2]*[1] / ([1] * [1] * [1] * [yr])
    emissions = (
        (airborne_emissions_new - np.sum(gas_boxes_old*decay_factor, axis=GAS_BOX_AXIS, keepdims=True)) /
        (np.sum(
            partition_fraction / decay_rate * ( 1. - decay_factor ) * timestep,
            axis=GAS_BOX_AXIS, keepdims=True)
        )
    )

    # [GtCO2] = [yr] * [GtCO2/yr] * [1] / [1] * [1] + [GtCO2] * [1]
    gas_boxes_new = timestep * emissions * partition_fraction * 1/decay_rate * ( 1. - decay_factor ) + gas_boxes_old * decay_factor

    emissions_out = emissions + natural_emissions_adjustment

    return emissions_out, gas_boxes_new, airborne_emissions_new
