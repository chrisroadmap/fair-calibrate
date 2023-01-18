# put imports outside: we don't have a lot of overhead here, and it looks nicer.
import os
import warnings

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")


def run_fair(cfg):
    scenarios = ["ssp245"]
    batch_start = cfg["batch_start"]
    batch_end = cfg["batch_end"]
    batch_size = batch_end - batch_start

    species, properties = read_properties()

    da_emissions = xr.load_dataarray(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
        "ssp_emissions_1750-2500.nc"
    )

    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2101, 1)
    f.define_scenarios(scenarios)
    f.define_configs(list(range(batch_start, batch_end)))
    f.define_species(species, properties)
    f.allocate()

    trend_shape = np.ones(352)
    trend_shape[:271] = np.linspace(0, 1, 271)

    da = da_emissions.loc[dict(config="unspecified", scenario="ssp245")][:351, ...]
    fe = da.expand_dims(dim=["scenario", "config"], axis=(1, 2))
    f.emissions = fe.drop("config") * np.ones((1, 1, batch_size, 1))

    # solar and volcanic forcing
    fill(
        f.forcing,
        cfg["volcanic_forcing"][:, None, None] * cfg["scaling_Volcanic"],
        specie="Volcanic",
    )
    fill(
        f.forcing,
        cfg["solar_forcing"][:, None, None] * cfg["scaling_solar_amplitude"]
        + trend_shape[:, None, None] * cfg["scaling_solar_trend"],
        specie="Solar",
    )

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
    fill(f.climate_configs["sigma_eta"], cfg["sigma_eta"])
    fill(f.climate_configs["sigma_xi"], cfg["sigma_xi"])
    fill(f.climate_configs["seed"], cfg["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], cfg["forcing_4co2"])

    # species level
    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs["iirf_0"], cfg["iirf_0"], specie="CO2")
    fill(f.species_configs["iirf_airborne"], cfg["iirf_airborne"], specie="CO2")
    fill(f.species_configs["iirf_uptake"], cfg["iirf_uptake"], specie="CO2")
    fill(f.species_configs["iirf_temperature"], cfg["iirf_temperature"], specie="CO2")

    # methane lifetime baseline - should be imported from calibration
    fill(f.species_configs["unperturbed_lifetime"], 10.11702748, specie="CH4")

    # emissions adjustments for N2O and CH4
    fill(f.species_configs["baseline_emissions"], 19.019783117809567, specie="CH4")
    fill(f.species_configs["baseline_emissions"], 0.08602230754, specie="N2O")

    # aerosol indirect
    fill(f.species_configs["aci_scale"], cfg["beta"])
    fill(f.species_configs["aci_shape"], cfg["shape_so2"], specie="Sulfur")
    fill(f.species_configs["aci_shape"], cfg["shape_bc"], specie="BC")
    fill(f.species_configs["aci_shape"], cfg["shape_oc"], specie="OC")

    # forcing scaling
    fill(f.species_configs["forcing_scale"], cfg["scaling_CO2"], specie="CO2")
    fill(f.species_configs["forcing_scale"], cfg["scaling_CH4"], specie="CH4")
    fill(f.species_configs["forcing_scale"], cfg["scaling_N2O"], specie="N2O")
    # fill(f.species_configs['forcing_scale'], cfg['scaling_minorGHG'], specie='CO2')
    fill(
        f.species_configs["forcing_scale"],
        cfg["scaling_stwv"],
        specie="Stratospheric water vapour",
    )
    fill(
        f.species_configs["forcing_scale"], cfg["scaling_contrails"], specie="Contrails"
    )
    fill(
        f.species_configs["forcing_scale"],
        cfg["scaling_lapsi"],
        specie="Light absorbing particles on snow and ice",
    )
    fill(f.species_configs["forcing_scale"], cfg["scaling_landuse"], specie="Land use")

    for specie in [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1202",
        "Halon-1301",
        "Halon-2402",
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ]:
        fill(f.species_configs["forcing_scale"], cfg["scaling_minorGHG"], specie=specie)

    # aerosol radiation interactions
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_BC"], specie="BC")
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_CH4"], specie="CH4")
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_CO"], specie="CO")
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_N2O"], specie="N2O")
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_NH3"], specie="NH3")
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_NOx"], specie="NOx")
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_OC"], specie="OC")
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        cfg["ari_Sulfur"],
        specie="Sulfur",
    )
    fill(f.species_configs["erfari_radiative_efficiency"], cfg["ari_VOC"], specie="VOC")
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        cfg["ari_EESC"],
        specie="Equivalent effective stratospheric chlorine",
    )

    # Ozone
    fill(
        f.species_configs["ozone_radiative_efficiency"], cfg["ozone_CH4"], specie="CH4"
    )
    fill(
        f.species_configs["ozone_radiative_efficiency"], cfg["ozone_N2O"], specie="N2O"
    )
    fill(f.species_configs["ozone_radiative_efficiency"], cfg["ozone_CO"], specie="CO")
    fill(
        f.species_configs["ozone_radiative_efficiency"], cfg["ozone_VOC"], specie="VOC"
    )
    fill(
        f.species_configs["ozone_radiative_efficiency"], cfg["ozone_NOx"], specie="NOx"
    )
    fill(
        f.species_configs["ozone_radiative_efficiency"],
        cfg["ozone_EESC"],
        specie="Equivalent effective stratospheric chlorine",
    )

    # tune down volcanic efficacy
    fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")

    # CO2 in 1750
    fill(f.species_configs["baseline_concentration"], cfg["CO2_1750"], specie="CO2")

    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    return (
        f.temperature[100:, 0, :, 0],
        f.ocean_heat_content_change[268:270, 0, :].mean(axis=0)
        - f.ocean_heat_content_change[221:223, 0, :].mean(axis=0),
        f.concentration[264, 0, :, 2],
        np.average(
            f.forcing[255:266, 0, :, 56],
            weights=np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]),
            axis=0,
        ),
        np.average(
            f.forcing[255:266, 0, :, 57],
            weights=np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]),
            axis=0,
        ),
        f.ebms.ecs,
        f.ebms.tcr,
    )
