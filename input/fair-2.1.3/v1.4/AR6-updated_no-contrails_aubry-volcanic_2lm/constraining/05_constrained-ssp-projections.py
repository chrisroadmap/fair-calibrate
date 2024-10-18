#!/usr/bin/env python
# coding: utf-8

"""Run constrained projections for SSPs"""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

pl.switch_backend("agg")

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

print("Running SSP scenarios...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

scenarios = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
]

df_solar = pd.read_csv(
    "../../../../../data/forcing/solar_erf_timebounds.csv", index_col="year"
)
df_volcanic = pd.read_csv(
    "../../../../../data/forcing/volcanic_ERF_aubry_1750-2101_timebounds_rebased_1850-2014.csv",
    index_col="Time",
)

solar_forcing = np.zeros(551)
volcanic_forcing = np.zeros(551)
volcanic_forcing[:352] = df_volcanic["erf"].loc[1750:2101].values
solar_forcing = df_solar["erf"].loc[1750:2300].values

df_methane = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "CH4_lifetime.csv",
    index_col=0,
)
df_configs = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters.csv",
    index_col=0,
)
valid_all = df_configs.index

trend_shape = np.ones(551)
trend_shape[:271] = np.linspace(0, 1, 271)

f = FAIR(ch4_method="Thornhill2021", n_layers=2)
f.define_time(1750, 2300, 1)
f.define_scenarios(scenarios)
f.define_configs(valid_all)
species, properties = read_properties()
species.remove("Halon-1202")
species.remove("NOx aviation")
species.remove("Contrails")
f.define_species(species, properties)
f.allocate()

# run with harmonized emissions
da_emissions = xr.load_dataarray(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssp_emissions_1750-2500.nc"
)

da = da_emissions.loc[dict(config="unspecified")][:550, ...]
fe = da.expand_dims(dim=["config"], axis=(2))
f.emissions = fe.drop("config") * np.ones((1, 1, output_ensemble_size, 1))

# solar and volcanic forcing
fill(
    f.forcing,
    volcanic_forcing[:, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    solar_forcing[:, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze(),
    specie="Solar",
)

# climate response
fill(
    f.climate_configs["ocean_heat_capacity"],
    df_configs.loc[:, "clim_c1":"clim_c2"].values,
)
fill(
    f.climate_configs["ocean_heat_transfer"],
    df_configs.loc[:, "clim_kappa1":"clim_kappa2"].values,
)  # not massively robust, since relies on kappa1, kappa2, kappa3 being in adjacent cols
fill(
    f.climate_configs["deep_ocean_efficacy"],
    df_configs["clim_epsilon"].values.squeeze(),
)
fill(
    f.climate_configs["gamma_autocorrelation"],
    df_configs["clim_gamma"].values.squeeze(),
)
fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
fill(f.climate_configs["seed"], df_configs["seed"])
fill(f.climate_configs["stochastic_run"], True)
fill(f.climate_configs["use_seed"], True)
fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])

# species level
f.fill_species_configs()

# carbon cycle
fill(f.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(), specie="CO2")
fill(
    f.species_configs["iirf_airborne"],
    df_configs["cc_rA"].values.squeeze(),
    specie="CO2",
)
fill(
    f.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(), specie="CO2"
)
fill(
    f.species_configs["iirf_temperature"],
    df_configs["cc_rT"].values.squeeze(),
    specie="CO2",
)

# aerosol indirect
fill(f.species_configs["aci_scale"], df_configs["aci_beta"].values.squeeze())
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_so2"].values.squeeze(),
    specie="Sulfur",
)
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_bc"].values.squeeze(),
    specie="BC",
)
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_oc"].values.squeeze(),
    specie="OC",
)

# methane lifetime baseline and sensitivity
fill(
    f.species_configs["unperturbed_lifetime"],
    df_methane.loc["historical_best", "base"],
    specie="CH4",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "CH4"],
    specie="CH4",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "N2O"],
    specie="N2O",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "VOC"],
    specie="VOC",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "NOx"],
    specie="NOx",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "HC"],
    specie="Equivalent effective stratospheric chlorine",
)
fill(
    f.species_configs["lifetime_temperature_sensitivity"],
    df_methane.loc["historical_best", "temp"],
)

# emissions adjustments for N2O and CH4
fill(f.species_configs["baseline_emissions"], 19.019783117809567, specie="CH4")
fill(f.species_configs["baseline_emissions"], 0.08602230754, specie="N2O")
fill(f.species_configs["baseline_emissions"], 19.423526730206152, specie="NOx")

# aerosol direct
for specie in [
    "BC",
    "CH4",
    "N2O",
    "NH3",
    "NOx",
    "OC",
    "Sulfur",
    "VOC",
    "Equivalent effective stratospheric chlorine",
]:
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        df_configs[f"ari_{specie}"],
        specie=specie,
    )

# forcing scaling
for specie in [
    "CO2",
    "CH4",
    "N2O",
    "Stratospheric water vapour",
    "Light absorbing particles on snow and ice",
    "Land use",
]:
    fill(
        f.species_configs["forcing_scale"],
        df_configs[f"fscale_{specie}"].values.squeeze(),
        specie=specie,
    )

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
    fill(
        f.species_configs["forcing_scale"],
        df_configs["fscale_minorGHG"].values.squeeze(),
        specie=specie,
    )

# ozone
for specie in [
    "CH4",
    "N2O",
    "Equivalent effective stratospheric chlorine",
    "CO",
    "VOC",
    "NOx",
]:
    fill(
        f.species_configs["ozone_radiative_efficiency"],
        df_configs[f"o3_{specie}"],
        specie=specie,
    )

# initial condition of CO2 concentration (but not baseline for forcing calculations)
fill(
    f.species_configs["baseline_concentration"],
    df_configs["cc_co2_concentration_1750"].values.squeeze(),
    specie="CO2",
)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run(progress=progress)

fancy_titles = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp434": "SSP4-3.4",
    "ssp460": "SSP4-6.0",
    "ssp534-over": "SSP5-3.4-overshoot",
    "ssp585": "SSP5-8.5",
}

ar6_colors = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534-over": "#92397a",
    "ssp585": "#980002",
}

df_gmst = pd.read_csv("../../../../../data/forcing/IGCC_GMST_1850-2022.csv")
gmst = df_gmst["gmst"].values

if plots:
    fig, ax = pl.subplots(2, 4, figsize=(18 / 2.54, 8 / 2.54))
    for i in range(8):
        ax[i // 4, i % 4].fill_between(
            np.arange(1750.5, 2301),
            np.min(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                axis=1,
            ),
            np.max(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                axis=1,
            ),
            color=ar6_colors[scenarios[i]],
            alpha=0.2,
            lw=0,
        )
        ax[i // 4, i % 4].fill_between(
            np.arange(1750.5, 2301),
            np.percentile(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                5,
                axis=1,
            ),
            np.percentile(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                95,
                axis=1,
            ),
            color=ar6_colors[scenarios[i]],
            alpha=0.2,
            lw=0,
        )
        ax[i // 4, i % 4].fill_between(
            np.arange(1750.5, 2301),
            np.percentile(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                16,
                axis=1,
            ),
            np.percentile(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                84,
                axis=1,
            ),
            color=ar6_colors[scenarios[i]],
            alpha=0.2,
            lw=0,
        )
        ax[i // 4, i % 4].plot(
            np.arange(1750.5, 2301),
            np.median(
                f.temperature[:, i, :, 0]
                - f.temperature[100:151, i, :, 0].mean(axis=0),
                axis=1,
            ),
            color=ar6_colors[scenarios[i]],
            lw=1,
        )
        ax[i // 4, i % 4].plot(np.arange(1850.5, 2023), gmst, color="k", lw=1)
        ax[i // 4, i % 4].set_xlim(1950, 2200)
        ax[i // 4, i % 4].set_ylim(-1, 10)
        ax[i // 4, i % 4].axhline(0, color="k", ls=":", lw=0.5)
        ax[i // 4, i % 4].set_title(fancy_titles[scenarios[i]])

    ax[0, 0].set_ylabel("°C since 1850-1900")
    ax[1, 0].set_ylabel("°C since 1850-1900")

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_ssp_temperatures.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_ssp_temperatures.pdf"
    )
    pl.close()

# # Temperature diffs w.r.t. 1995-2014
# Future periods are 2021-2040, 2041-2060, 2081-2100. Values are 5th, 50th, 95th

# as temperatures are timebounds, we should weight first and last by 0.5

weight_20yr = np.ones(21)
weight_20yr[0] = 0.5
weight_20yr[-1] = 0.5

weight_51yr = np.ones(52)
weight_51yr[0] = 0.5
weight_51yr[-1] = 0.5

temp_model_19952014 = np.zeros((15, 3))
temp_model_18501900 = np.zeros((15, 3))

periodmap = {0: slice(271, 292), 1: slice(291, 312), 2: slice(331, 352)}
for irow in range(15):
    scenmap = irow // 3
    if scenmap == 4:
        scenmap = 7
    temp_model_19952014[irow, :] = np.percentile(
        np.average(
            f.temperature[periodmap[irow % 3], scenmap, :, 0],
            weights=weight_20yr,
            axis=0,
        )
        - np.average(
            f.temperature[245:266, scenmap, :, 0], weights=weight_20yr, axis=0
        ),
        (5, 50, 95),
    )
    temp_model_18501900[irow, :] = np.percentile(
        np.average(
            f.temperature[periodmap[irow % 3], scenmap, :, 0],
            weights=weight_20yr,
            axis=0,
        )
        - np.average(
            f.temperature[100:152, scenmap, :, 0], weights=weight_51yr, axis=0
        ),
        (5, 50, 95),
    )

print("Anomalies rel. 1995-2014:")
print((temp_model_19952014))
print()
print("Anomalies rel. 1850-1900:")
print((temp_model_18501900))
print()
print(
    "Methane forcing 2019:",
    np.percentile(f.forcing[269:271, 2, :, 3].mean(axis=0), (5, 50, 95)),
)
print(
    "Methane concentration 2019:",
    np.percentile(f.concentration[269:271, 2, :, 3].mean(axis=0), (5, 50, 95)),
)
print(
    "WMGHG forcing 2019:",
    np.percentile(
        (
            f.forcing[269:271, 2, :, 2:5].sum(axis=2)
            + f.forcing[269:271, 2, :, 12:52].sum(axis=2)
        ).mean(axis=0),
        (5, 50, 95),
    ),
)


rcmip_concentration_file = pooch.retrieve(
    url=("doi:10.5281/zenodo.4589756/" "rcmip-concentrations-annual-means-v5-1-0.csv"),
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
)


df_conc = pd.read_csv(rcmip_concentration_file)
conc_ch4 = {}
conc_co2 = {}
for scenario in scenarios:
    conc_ch4[scenario] = (
        df_conc.loc[
            (df_conc["Scenario"] == scenario)
            & (df_conc["Variable"].str.endswith("|CH4"))
            & (df_conc["Region"] == "World"),
            "1750":"2500",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    conc_co2[scenario] = (
        df_conc.loc[
            (df_conc["Scenario"] == scenario)
            & (df_conc["Variable"].str.endswith("|CO2"))
            & (df_conc["Region"] == "World"),
            "1750":"2500",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )


if plots:
    pl.plot(np.percentile(f.concentration[:, 2, :, 3], (50), axis=1))
    pl.plot(conc_ch4["ssp245"][:], color="k")
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/ch4_ssp245.png"
    )
    pl.close()

    pl.plot(
        df_methane.loc["historical_best", "base"]
        * np.percentile(f.alpha_lifetime[:, 2, :, 3], (50), axis=1)
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ch4lifetime_ssp245.png"
    )
    pl.close()

    pl.plot(np.percentile(f.concentration[:, 2, :, 60], (50), axis=1))
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/eesc_ssp245.png"
    )
    pl.close()

    pl.plot(
        np.arange(1750, 2301),
        np.percentile(f.concentration[:, 7, :, 2], (50), axis=1),
        label="fair2.1 median",
    )
    pl.plot(
        np.arange(1750, 2301),
        conc_co2["ssp585"][:551],
        color="k",
        label="SSP historical",
    )
    pl.legend()
    pl.xlim(1750, 2300)
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "co2_historical.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1950, 2101),
        np.percentile(
            f.toa_imbalance[200:351, 2, :],
            5,
            axis=1,
        ),
        np.percentile(
            f.toa_imbalance[200:351, 2, :],
            95,
            axis=1,
        ),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1950, 2101),
        np.median(
            f.toa_imbalance[200:351, 2, :],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "toa_imbalance_ssp245.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "toa_imbalance_ssp245.pdf"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(
            f.forcing[:351, 2, :, 2:5].sum(axis=2)
            + f.forcing[:351, 2, :, 12:52].sum(axis=2),
            5,
            axis=1,
        ),
        np.percentile(
            f.forcing[:351, 2, :, 2:5].sum(axis=2)
            + f.forcing[:351, 2, :, 12:52].sum(axis=2),
            95,
            axis=1,
        ),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 2:5].sum(axis=2)
            + f.forcing[:351, 2, :, 12:52].sum(axis=2),
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ghg_forcing_ssp245.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 2, :, 54:56].sum(axis=2), 5, axis=1),
        np.percentile(f.forcing[:351, 2, :, 54:56].sum(axis=2), 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 54:56].sum(axis=2),
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "aerosol_forcing_ssp245.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 2, :, 2], 5, axis=1),
        np.percentile(f.forcing[:351, 2, :, 2], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 2],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "co2_forcing_ssp245.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 2, :, 56], 5, axis=1),
        np.percentile(f.forcing[:351, 2, :, 56], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 56],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ozone_ssp245.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 2, :, 57], 5, axis=1),
        np.percentile(f.forcing[:351, 2, :, 57], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 57],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "lapsi_ssp245.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 2, :, 58], 5, axis=1),
        np.percentile(f.forcing[:351, 2, :, 58], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 58],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "stratH2O_ssp245.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 2, :, 59], 5, axis=1),
        np.percentile(f.forcing[:351, 2, :, 59], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 2, :, 59],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "landuse_ssp245.png"
    )
    pl.close()
