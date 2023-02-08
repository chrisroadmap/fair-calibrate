
"""Calibrate nitrous oxide lifetime from new emissions data."""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
from scipy.interpolate import interp1d
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
import xarray as xr

load_dotenv()

print("Calibrating nitrous oxide lifetime...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")

# ## Temperature data
# Use observations 1850-2019

df_temp = pd.read_csv("../../../../../data/forcing/AR6_GMST.csv")
gmst = np.zeros(270)
gmst[100:270] = df_temp["gmst"].values[:-1]

# ## Get emissions and concentrations
# this time from observations and PRIMAP
df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "primap_ceds_gfed_1750-2021.csv",
    index_col=0
)

df_conc = pd.read_csv(
    "../../../../../data/concentrations/LLGHG_history_AR6_v9_for_archive.csv",
    index_col=0
)

# We presume concentrations are mid-year from IPCC. That's fine for calibration.
# When running FaIR, remember concentrations are on time bounds.

input = {}
hc_input = {}

conc_species = ["N2O"]
output_years = np.arange(1750, 2020)
conc_years = np.concatenate(([1750], np.arange(1850, 2020)))
for species in conc_species:
    temp_input = df_conc.loc[:, species].values
    f = interp1d(conc_years, temp_input)
    input[species] = f(output_years)

input["temp"] = gmst


# put this into a simple one box model
def one_box(
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

    effective_lifetime = (alpha_lifetime * lifetime)
    decay_rate = timestep / (effective_lifetime)
    decay_factor = np.exp(-decay_rate)
    gas_boxes_new = (
        partition_fraction
        * (emissions - natural_emissions_adjustment)
        * 1
        / decay_rate
        * (1 - decay_factor)
        * timestep
        + gas_boxes_old * decay_factor
    )
    airborne_emissions_new = gas_boxes_new
    concentration_out = (
        pre_industrial_concentration
        + burden_per_emission * airborne_emissions_new
    )
    return concentration_out, gas_boxes_new, airborne_emissions_new


emis_n2o = df_emis.loc[df_emis["Variable"]=='Emissions|N2O', "1750":"2019"].values.squeeze()
burden_per_emission = 1 / (5.1352e18 / 1e18 * 44.013 / 28.97)
partition_fraction = 1
pre_industrial_concentration = 270.1
natural_emissions_adjustment = emis_n2o[0]

conc_n2o = {}

# ## Step 3
#
# Find least squares sensible historical fit
# invect = np.array(
#     [input["CH4"], input["NOx"], input["VOC"], input["HC"], input["N2O"], input["temp"]]
# )


def fit_precursors(emis, rbase):
    conc_n2o = np.zeros(271)
    gas_boxes = 0
    airborne_emissions = 0
    conc_n2o[0] = pre_industrial_concentration

    for i in range(270):
        conc_n2o[i+1], gas_boxes, airborne_emissions = one_box(
            emis[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            rbase,
            1,
            partition_fraction,
            pre_industrial_concentration,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )

    # convert bounds to midyear
    conc_n2o = (conc_n2o[1:] + conc_n2o[:-1])/2
    return conc_n2o


p, cov = scipy.optimize.curve_fit(
    fit_precursors,
    emis_n2o,
    input["N2O"][:270],
)

print(p[0])


conc_n2o["best_fit"] = np.zeros(271)
gas_boxes = 0
airborne_emissions = 0
conc_n2o["best_fit"][0] = pre_industrial_concentration
for i in range(270):
    conc_n2o["best_fit"][i+1], gas_boxes, airborne_emissions = one_box(
        emis_n2o[i],
        gas_boxes,
        airborne_emissions,
        burden_per_emission,
        p[0],
        1,
        partition_fraction,
        pre_industrial_concentration,
        timestep=1,
        natural_emissions_adjustment=natural_emissions_adjustment,
    )

if plots:
    fig, ax = pl.subplots(figsize=(3.5, 3.5))
    ax.plot(
        np.arange(1750, 2021), conc_n2o["best_fit"], color="0.5", label="Best fit"
    )
    ax.plot(
        np.arange(1750.5, 2020), input["N2O"], color="k", label="observations: AR6"
    )
    ax.set_ylabel("ppb")
    ax.set_xlim(1750, 2020)
    ax.legend(frameon=False)
    ax.set_title("N$_2$O concentration")
    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "n2o_calibrations.png"
    )
    pl.close()
