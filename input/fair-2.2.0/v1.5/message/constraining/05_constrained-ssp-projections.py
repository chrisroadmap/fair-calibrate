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
    "baseline",
]

df_solar = pd.read_csv(
    "../../../../../data/forcing/solar_erf_timebounds.csv", index_col="year"
)
df_volcanic = pd.read_csv(
    "../../../../../data/forcing/volcanic_ERF_1750-2101_timebounds.csv",
    index_col="timebounds",
)
fair_species_configs_1_5_0 = "../../../../../data/calibrations/species_configs_properties_calibration1.5.0.csv"
fair_parameters_1_5_0 = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/calibrated_constrained_parameters.csv'

df_configs = pd.read_csv(fair_parameters_1_5_0, index_col=0)

solar_forcing = np.zeros(357)
volcanic_forcing = np.zeros(357)
volcanic_forcing[:352] = df_volcanic["erf"].loc[1750:2101].values
solar_forcing = df_solar["erf"].loc[1750:2106].values

valid_all = df_configs.index


f = FAIR(ch4_method="Thornhill2021")
f.define_time(1750, 2106, 1)
f.define_scenarios(scenarios)
f.define_configs(valid_all)
species, properties = read_properties(fair_species_configs_1_5_0)
f.define_species(species, properties)
f.allocate()

# run with scenario emissions
f.fill_from_csv(
    emissions_file='../../../../../data/emissions/message-baseline-2020.csv',
)

# solar and volcanic forcing
fill(
    f.forcing,
    volcanic_forcing[:, None, None] * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    solar_forcing[:, None, None] * df_configs["forcing_scale[Solar]"].values.squeeze(),
    specie="Solar",
)

f.fill_species_configs(fair_species_configs_1_5_0)
f.override_defaults(fair_parameters_1_5_0)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)
initialise(f.ocean_heat_content_change, 0)

f.run(progress=progress)

fancy_titles = {
    "baseline": "Baseline",
}

ar6_colors = {
    "baseline": "#f69320",
}

df_gmst = pd.read_csv("../../../../../data/forcing/IGCC_GMST_1850-2023.csv")
gmst = df_gmst["gmst"].values

if plots:
    i = 0
    fig, ax = pl.subplots(figsize=(9 / 2.54, 9 / 2.54))
    ax.fill_between(
        np.arange(1750, 2107),
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
    ax.fill_between(
        np.arange(1750, 2107),
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
    ax.fill_between(
        np.arange(1750, 2107),
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
    ax.plot(
        np.arange(1750, 2107),
        np.median(
            f.temperature[:, i, :, 0]
            - f.temperature[100:151, i, :, 0].mean(axis=0),
            axis=1,
        ),
        color=ar6_colors[scenarios[i]],
        lw=1,
    )
    ax.plot(np.arange(1850.5, 2024), gmst, color="k", lw=1)
    ax.set_xlim(1950, 2106)
    ax.set_ylim(-1, 5)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_title(fancy_titles[scenarios[i]])

    ax.set_ylabel("°C since 1850-1900")

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
for irow in range(3):
    scenmap = irow // 3
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
    np.percentile(f.forcing[269:271, 0, :, 3].mean(axis=0), (5, 50, 95)),
)
print(
    "Methane concentration 2019:",
    np.percentile(f.concentration[269:271, 0, :, 3].mean(axis=0), (5, 50, 95)),
)
print(
    "WMGHG forcing 2019:",
    np.percentile(
        (
            f.forcing[269:271, 0, :, 2:5].sum(axis=2)
            + f.forcing[269:271, 0, :, 12:21].sum(axis=2)
        ).mean(axis=0),
        (5, 50, 95),
    ),
)


if plots:
    pl.plot(np.percentile(f.concentration[:, 0, :, 3], (50), axis=1))
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/ch4_baseline.png"
    )
    pl.close()

    pl.plot(
        9.85
        * np.percentile(f.alpha_lifetime[:, 0, :, 3], (50), axis=1)
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ch4lifetime_baseline.png"
    )
    pl.close()

    pl.plot(
        np.arange(1750, 2107),
        np.percentile(f.concentration[:, 0, :, 2], (50), axis=1),
        label="fair2.2 median",
    )
    pl.legend()
    pl.xlim(1750, 2300)
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "co2_historical_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1950, 2101),
        np.percentile(
            f.toa_imbalance[200:351, 0, :],
            5,
            axis=1,
        ),
        np.percentile(
            f.toa_imbalance[200:351, 0, :],
            95,
            axis=1,
        ),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1950, 2101),
        np.median(
            f.toa_imbalance[200:351, 0, :],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "toa_imbalance_baseline.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "toa_imbalance_baseline.pdf"
    )
    pl.close()

    pl.fill_between(
        np.arange(1950, 2101),
        np.percentile(
            f.forcing_sum[200:351, 0, :],
            5,
            axis=1,
        ),
        np.percentile(
            f.forcing_sum[200:351, 0, :],
            95,
            axis=1,
        ),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1950, 2101),
        np.median(
            f.forcing_sum[200:351, 0, :],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "forcing_sum_baseline.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "forcing_sum_baseline.pdf"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(
            f.forcing[:351, 0, :, 2:5].sum(axis=2)
            + f.forcing[:351, 0, :, 12:21].sum(axis=2),
            5,
            axis=1,
        ),
        np.percentile(
            f.forcing[:351, 0, :, 2:5].sum(axis=2)
            + f.forcing[:351, 0, :, 12:21].sum(axis=2),
            95,
            axis=1,
        ),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 2:5].sum(axis=2)
            + f.forcing[:351, 0, :, 12:21].sum(axis=2),
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ghg_forcing_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 0, :, 23:25].sum(axis=2), 5, axis=1),
        np.percentile(f.forcing[:351, 0, :, 23:25].sum(axis=2), 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 23:25].sum(axis=2),
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "aerosol_forcing_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 0, :, 2], 5, axis=1),
        np.percentile(f.forcing[:351, 0, :, 2], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 2],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "co2_forcing_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 0, :, 25], 5, axis=1),
        np.percentile(f.forcing[:351, 0, :, 25], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 25],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ozone_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 0, :, 26], 5, axis=1),
        np.percentile(f.forcing[:351, 0, :, 26], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 26],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "lapsi_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 0, :, 27], 5, axis=1),
        np.percentile(f.forcing[:351, 0, :, 27], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 27],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "stratH2O_baseline.png"
    )
    pl.close()

    pl.fill_between(
        np.arange(1750, 2101),
        np.percentile(f.forcing[:351, 0, :, 28], 5, axis=1),
        np.percentile(f.forcing[:351, 0, :, 28], 95, axis=1),
        color="k",
        alpha=0.3,
    )
    pl.plot(
        np.arange(1750, 2101),
        np.median(
            f.forcing[:351, 0, :, 28],
            axis=1,
        ),
        color="k",
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "landuse_baseline.png"
    )
    pl.close()
