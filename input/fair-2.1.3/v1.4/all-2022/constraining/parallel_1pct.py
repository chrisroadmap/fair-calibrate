# put imports outside: we don't have a lot of overhead here, and it looks nicer.
import os
import warnings

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
from scipy.interpolate import interp1d

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")


def run_fair(cfg):
    scenarios = ["1pctCO2"]
    batch_start = cfg["batch_start"]
    batch_end = cfg["batch_end"]
    batch_size = batch_end - batch_start

    species, properties = read_properties()

    da_concentration = xr.load_dataarray(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "concentration/1pctCO2_concentration_1850-1990.nc"
    )

    f = FAIR()
    f.define_time(1850, 1990, 1)
    f.define_scenarios(scenarios)
    species = ["CO2", "CH4", "N2O"]
    properties = {
        "CO2": {
            "type": "co2",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "CH4": {
            "type": "ch4",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "N2O": {
            "type": "n2o",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
    }
    f.define_configs(list(range(batch_start, batch_end)))
    f.define_species(species, properties)
    f.allocate()

    da = da_concentration.loc[dict(config="unspecified", scenario="1pctCO2")]
    fe = da.expand_dims(dim=["scenario", "config"], axis=(1, 2))
    f.concentration = fe.drop("config") * np.ones((1, 1, batch_size, 1))

    # climate response
    fill(
        f.climate_configs["ocean_heat_capacity"],
        np.array([cfg["c1"], cfg["c2"], cfg["c3"]]).T,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        np.array([cfg["kappa1"], cfg["kappa2"], cfg["kappa3"]]).T,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], cfg["epsilon"])
    fill(f.climate_configs["gamma_autocorrelation"], cfg["gamma"])
    fill(f.climate_configs["stochastic_run"], False)
    fill(f.climate_configs["forcing_4co2"], cfg["forcing_4co2"])

    # species level
    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs["iirf_0"], cfg["iirf_0"], specie="CO2")
    fill(f.species_configs["iirf_airborne"], cfg["iirf_airborne"], specie="CO2")
    fill(f.species_configs["iirf_uptake"], cfg["iirf_uptake"], specie="CO2")
    fill(f.species_configs["iirf_temperature"], cfg["iirf_temperature"], specie="CO2")

    # forcing scaling
    fill(f.species_configs["forcing_scale"], cfg["scaling_CO2"], specie="CO2")
    fill(f.species_configs["forcing_scale"], cfg["scaling_CH4"], specie="CH4")
    fill(f.species_configs["forcing_scale"], cfg["scaling_N2O"], specie="N2O")

    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(f.species_configs["baseline_concentration"], 284.3169988, specie="CO2")
    fill(f.species_configs["baseline_concentration"], 808.2490285, specie="CH4")
    fill(f.species_configs["baseline_concentration"], 273.021047, specie="N2O")

    fill(
        f.species_configs["forcing_reference_concentration"], 284.3169988, specie="CO2"
    )
    fill(
        f.species_configs["forcing_reference_concentration"], 808.2490285, specie="CH4"
    )
    fill(f.species_configs["forcing_reference_concentration"], 273.021047, specie="N2O")

    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    # interpolate warming at 1000 GtC
    t1000 = np.ones(batch_size) * np.nan
    ttco2 = 1000 * 44.009 / 12.011
    for ibatch in range(batch_size):
        interpolator = interp1d(
            f.cumulative_emissions[:, 0, ibatch, 0], f.temperature[:, 0, ibatch, 0]
        )
        t1000[ibatch] = interpolator(ttco2)

    return (
        np.array((f.temperature[70, 0, :, 0], f.temperature[140, 0, :, 0])),
        np.array((f.airborne_fraction[70, 0, :, 0], f.airborne_fraction[140, 0, :, 0])),
        np.array(t1000),
    )
