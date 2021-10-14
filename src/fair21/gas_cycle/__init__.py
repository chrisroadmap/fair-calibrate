"""
Module containing gas cycle functions
"""

from ..defaults.gases import iirf_horizon

import numpy as np

from ..constants.gases import burden_per_emission, lifetime
from ..defaults.gases import (
    pre_industrial_concentration,
    partition_fraction,
    iirf_horizon
)

def calculate_g(
    lifetime,
    partition_fraction=1,
    iirf_horizon=iirf_horizon,
    
):
    """Calculate the `g` components of the gas cycle.
    
    See Leach et al. (2021), eq. (5)
    
    Inputs
    ------
    lifetime : float
        atmospheric burden lifetime of the greenhouse gas (yr).
    partition_fraction : float, default=1 or `np.ndarray` of float
        proportion of emissions of gas that go into each atmospheric box.
        Should be 1 or sum to 1 if array.
    iirf_horizon : float, default=100
        time horizon (yr) for time integrated impulse response function.
    
    Returns
    -------
    g0 : float
    g1 : float
    """
    
    g1 = np.sum(partition_fraction * lifetime * (1 - (1 + iirf_horizon/lifetime) * np.exp(-iirf_horizon/lifetime)))
    g0 = 1/(np.sinh(np.sum(partition_fraction*lifetime*(1 - np.exp(-iirf_horizon/lifetime)), axis=-1)/g1))
    return g0, g1

def calculate_alpha(
    cumulative_emissions,
    airborne_emissions,
    temperature,
    iirf_0,
    iirf_cumulative,
    iirf_temperature,
    iirf_airborne,
    g0,
    g1,
    iirf_max = 97.0,
):
    """
    Calculate greenhouse-gas time constant scaling factor.

    Inputs
    ------
        cumulative_emissions : float
            GtC cumulative emissions since pre-industrial.
        airborne_emissions : float
            GtC total emissions remaining in the atmosphere.
        temperature : float
            K temperature anomaly since pre-industrial.
        iirf_0 : float
            pre-industrial time-integrated airborne fraction.
        iirf_cumulative : float
            sensitivity of time-integrated airborne fraction with atmospheric
            carbon stock.
        iirf_temperature : float
            sensitivity of time-integrated airborne fraction with temperature
            anomaly.
        iirf_airborne : float
            sensitivity of time-integrated airborne fraction with airborne
            emissions.
        g0 : float
            parameter for alpha TODO: description
        g1 : float
            parameter for alpha TODO: description
        iirf_max : float
            maximum allowable value to time-integrated airborne fraction
    Returns
    -------
        alpha : float
            scaling factor for lifetimes
    """

    iirf = iirf_0 + iirf_cumulative * (cumulative_emissions-airborne_emissions) + iirf_temperature * temperature + iirf_airborne * airborne_emissions
    iirf = (iirf>iirf_max) * iirf_max + iirf * (iirf<iirf_max)
    alpha = g0 * np.sinh(iirf / g1)

    return alpha

def step_concentration_1box(
    emissions,
    airborne_emissions_old,
    lifetime,
    burden_per_emission,
    alpha_lifetime=1,
    pre_industrial_concentration=0,
    timestep=1
):
    """
    Calculates the concentrations from emissions for a generic greenhouse gas.

    Inputs
    ------
    emissions: float
        emissions in timestep.
    airborne_emissions_old : float
        The total airborne emissions at the beginning of the timestep. This is
        the concentrations above the pre-industrial control.
    lifetime : float or `np.ndarray`
        atmospheric burden lifetime of greenhouse gas
    burden_per_emission:
        how much atmospheric concentrations grow (e.g. in ppt) per unit (e.g.
        kt) emission.
    alpha_lifetime : float, default=1
        scaling factor for the default atmospheric lifetimes.
    pre_industrial_concentration : float, default=0
        pre-industrial concentration of the gas.
    timestep : float, default=1
        emissions timestep in years.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
        on the same time intervals: the airborne_emissions values are on time
        boundaries and these are averaged before being returned.

    Returns
    -------
    concentration_out : float
        greenhouse gas concentration at the centre of the timestep.
    airborne_emissions_new : float
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """
    decay_rate = timestep/(alpha_lifetime * lifetime)
    decay_factor = np.exp(-decay_rate)

    # Nick says: there shouldn't be a dt in the first decay rate
    # Chris says: there should, and there should be one here too. Emissions are a rate, e.g. Gt / yr
    airborne_emissions_new = emissions * 1 / decay_rate * (1 - decay_factor) * timestep + airborne_emissions_old * decay_factor

    concentration_out = pre_industrial_concentration + burden_per_emission * (airborne_emissions_new + airborne_emissions_old) / 2

    return concentration_out, airborne_emissions_new


def step_concentration_co2(
    emissions,
    gas_boxes_old,
    airborne_emissions_old,
    burden_per_emission=burden_per_emission['CO2'],
    alpha_lifetime=1,
    pre_industrial_concentration=pre_industrial_concentration['CO2'],  # put in a defaults module
    timestep=1,
    partition_fraction = partition_fraction['CO2'],
    lifetime = lifetime['CO2'],
):
    """
    Calculates the concentrations from emissions for CO2.

    Inputs
    ------
    emissions : float
        emissions in timestep.
    gas_boxes_old : `np.ndarray` of float
        carbon partition boxes at the end of the previous timestep.
    airborne_emissions_old : float
        The total airborne emissions at the beginning of the timestep. This is
        the concentrations above the pre-industrial control.
    burden_per_emission : float, default=`fair21.constants.burden_per_emission['CO2']`
        how much atmospheric concentrations grow (e.g. in ppm) per unit (e.g.
        Gt) emission.
    alpha_lifetime : float, default=1
        scaling factor for the default atmospheric lifetimes.
    pre_industrial_concentration : float, default=`fair21.defaults.gases.pre_industrial_concentration['CO2']`
        pre-industrial concentration of CO2.
    timestep : float, default=1
        emissions timestep in years.
    partition_fraction : float, default=fair21.defaults.gases.partition_fraction['CO2']`
        the partition fraction of emissions into each `gas_box`.
    lifetime : `np.ndarray` of float
        atmospheric burden lifetime of greenhouse gas.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
        on the same time intervals: the airborne_emissions values are on time
        boundaries and these are averaged before being returned.

    Returns
    -------
    concentration_out : float
        greenhouse gas concentration at the centre of the timestep.
    gas_boxes_new : `np.ndarray` of float
        carbon partition boxes at the end of the current timestep.
    airborne_emissions_new : float
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """
    decay_rate = timestep/(alpha_lifetime * lifetime)
    decay_factor = np.exp(-decay_rate)

    gas_boxes_new = partition_fraction * emissions * 1 / decay_rate * (1 - decay_factor) * timestep + gas_boxes_old * decay_factor
    airborne_emissions_new = np.sum(gas_boxes_new)

    concentration_out = pre_industrial_concentration + burden_per_emission * (airborne_emissions_new + airborne_emissions_old) / 2

    return concentration_out, gas_boxes_new, airborne_emissions_new
