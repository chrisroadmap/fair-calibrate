# put imports outside: we don't have a lot of overhead here, and it looks nicer.
import os
import warnings

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
from fair.forcing.ghg import meinshausen2020

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

    # hacky, but needed: add natural sources to methane and nitrous oxide emissions
    # need a fair code dev to make this part of the core code
    f.emissions.loc[dict(specie='CH4')] = f.emissions.loc[dict(specie='CH4')] + cfg["ch4_natural"]
    f.emissions.loc[dict(specie='N2O')] = f.emissions.loc[dict(specie='N2O')] + cfg["n2o_natural"]

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

    # methane lifetime baseline and sensitivity
    fill(f.species_configs["unperturbed_lifetime"], cfg["ch4_base"], specie="CH4")
    fill(
        f.species_configs["ch4_lifetime_chemical_sensitivity"],
        cfg["ch4_CH4"],
        specie="CH4",
    )
    fill(
        f.species_configs["ch4_lifetime_chemical_sensitivity"],
        cfg["ch4_N2O"],
        specie="N2O",
    )
    fill(
        f.species_configs["ch4_lifetime_chemical_sensitivity"],
        cfg["ch4_VOC"],
        specie="VOC",
    )
    fill(
        f.species_configs["ch4_lifetime_chemical_sensitivity"],
        cfg["ch4_NOx"],
        specie="NOx",
    )
    fill(
        f.species_configs["ch4_lifetime_chemical_sensitivity"],
        cfg["ch4_EESC"],
        specie="Equivalent effective stratospheric chlorine",
    )
    fill(f.species_configs["lifetime_temperature_sensitivity"], cfg["ch4_temp"])

    # N2O baseline lifetime
    fill(f.species_configs["unperturbed_lifetime"], cfg["n2o_base"], specie="N2O")

    # override for NOx with corrected emissions
    fill(f.species_configs["baseline_emissions"], 19.423526730206152, specie="NOx")

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

    # land use parameter needs rescaling
    fill(
        f.species_configs["land_use_cumulative_emissions_to_forcing"],
        -0.000236847,
        specie="CO2 AFOLU",
    )

    # CO2 in 1750, and baselines of zero for CH4 and N2O to take natural emissions
    fill(f.species_configs["baseline_concentration"], cfg["CO2_1750"], specie="CO2")
    fill(f.species_configs["baseline_concentration"], 0, specie="CH4")
    fill(f.species_configs["baseline_concentration"], 0, specie="N2O")

    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.concentration, 729.2, specie="CH4")
    initialise(f.concentration, 270.1, specie="N2O")
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # hack the methane and nitrous oxide to relax back to zero, not pre-industrial,
    # in the presence of balancing natural emissions
    initialise(f.airborne_emissions, 729.2 / (1 / (5.1352e18 / 1e18 * 16.043 / 28.97)), specie="CH4")
    initialise(f.airborne_emissions, 270.1 / (1 / (5.1352e18 / 1e18 * 44.013 / 28.97)), specie="N2O")
    fill(f.gas_partitions, np.array([1,0,0,0]) * 729.2 / (1 / (5.1352e18 / 1e18 * 16.043 / 28.97)), specie="CH4")
    fill(f.gas_partitions, np.array([1,0,0,0]) * 270.1 / (1 / (5.1352e18 / 1e18 * 44.013 / 28.97)), specie="N2O")

    # # the forcing reference concentration is overwritten to use zero, so we have to
    # # switch it back. Use Meinshausen 1750 or our PI values here?
    # fill(f.species_configs["forcing_reference_concentration"], 273.87, specie="N2O")
    # fill(f.species_configs["forcing_reference_concentration"], 731.41, specie="CH4")

    # recalculate the IIRF parameters for GHGs after changing base lifetimes, and switch
    # off the self-feedback for N2O which causes alpha_scaling and concentration to
    # go awry. The self-feedback is small, and a target for future recalibration
    f.calculate_g()
    f.calculate_iirf0()
    fill(f.species_configs["iirf_airborne"], 0, specie="N2O")

    # Then the CO2 parameters need to be overridden
    fill(f.species_configs["iirf_0"], cfg["iirf_0"], specie="CO2")
    fill(f.species_configs["iirf_airborne"], cfg["iirf_airborne"], specie="CO2")
    fill(f.species_configs["iirf_uptake"], cfg["iirf_uptake"], specie="CO2")
    fill(f.species_configs["iirf_temperature"], cfg["iirf_temperature"], specie="CO2")

    # define zero forcing as pre-industrial concentrations, not zero concentrations
    baseline = f.species_configs["baseline_concentration"][:, [2,3,4]]
    baseline[:, 1] = 729.2
    baseline[:, 2] = 270.1
    ghg_forcing_offset = meinshausen2020(
        baseline.data[None, None, ...],
        f.species_configs["forcing_reference_concentration"].data[None, None, :, [2,3,4]],
        f.species_configs["forcing_scale"].data[None, None, :, [2,3,4]] * (
            1 + f.species_configs["tropospheric_adjustment"].data[None, None, :, [2,3,4]]
        ),
        np.ones((1, 1, 1, 3)),
        0,
        1,
        2,
        [],
    )
    f.ghg_forcing_offset = np.zeros((1, 1) + f.species_configs["forcing_reference_concentration"].shape)
    f.ghg_forcing_offset[:, :, :, [2, 3, 4]] = ghg_forcing_offset[None, None, ...]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    import matplotlib.pyplot as pl
    pl.plot(f.concentration[:, 0, :, 3])
    pl.show()

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