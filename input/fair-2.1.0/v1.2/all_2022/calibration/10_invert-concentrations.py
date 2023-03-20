#!/usr/bin/env python
# coding: utf-8

"""Convert concentrations of minor GHGs to equivalent emissions."""

# In this calibration, we'll just take the species that appear in RCMIP otherwise
# there's nothing to harmonize to.

# Lifetime defaults are from RCMIP and in FaIR already.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.optimize
import scipy.stats
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR, __version__
from scipy.interpolate import interp1d

load_dotenv()

print("Calculating historical equivalent emissions...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")


f = FAIR()
f.define_time(1750, 2023, 1)
f.define_scenarios(["historical"])
f.define_configs(["historical"])
species = [
    'CFC-11',
    'CFC-12',
    'CFC-113',
    'CFC-114',
    'CFC-115',
    'HCFC-22',
    'HCFC-141b',
    'HCFC-142b',
    'CCl4',
    'CHCl3',
    'CH2Cl2',
    'CH3Cl',
    'CH3CCl3',
    'CH3Br',
    'Halon-1202',
    'Halon-1211',
    'Halon-1301',
    'Halon-2402',
    'CF4',
    'C2F6',
    'C3F8',
    'c-C4F8',
    'C4F10',
    'C5F12',
    'C6F14',
    'C7F16',
    'C8F18',
    'NF3',
    'SF6',
    'SO2F2',
    'HFC-125',
    'HFC-134a',
    'HFC-143a',
    'HFC-152a',
    'HFC-227ea',
    'HFC-23',
    'HFC-236fa',
    'HFC-245fa',
    'HFC-32',
    'HFC-365mfc',
    'HFC-4310mee',
]
properties = {
    
}




# Find least squares sensible historical fit using best estimate emissions and
# concentrations (not those from RCMIP)
df_conc_obs = pd.read_csv('../../../../../data/concentrations/ghg_concentrations_1750-2022.csv', index_col=0)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

input_obs = {}
input_obs['SF6'] = df_conc_obs['SF6'].values

x = np.arange(1750.5, 2023)
y = df_conc_obs['SF6'].values
f = interp1d(x, y)
input_obs['SF6'][1:] = f(np.arange(1751, 2023))
input_obs['SF6'][0] = 0

df_emis_obs = pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/primap_ceds_gfed_1750-2022.csv')
emis_obs = df_emis_obs.loc[df_emis_obs['Variable']=='Emissions|SF6', '1750':'2022'].values.squeeze()

# baselines are 1850! so "base" lifetime out is for 1750!
baseline_obs = {}
for species in ["SF6"]:
    baseline_obs[species] = input_obs[species][100]

burden_per_emission = 1 / (5.1352e18 / 1e18 * 146.06 / 28.97)
partition_fraction = 1
pre_industrial_concentration = 0#729.2
natural_emissions_adjustment = 0#emis_ch4[0]

def fit_precursors(x, rbase):
    conc_sf6 = np.zeros(273)
    gas_boxes = 0
    airborne_emissions = 0

    for i in range(273):
        conc_sf6[i], gas_boxes, airborne_emissions = one_box(
            emis_obs[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            rbase,
            1,#lifetime_scaling[i],
            partition_fraction,
            pre_industrial_concentration=0,
            timestep=1,
            natural_emissions_adjustment=0#natural_emissions_adjustment,
        )
    return conc_sf6


# natural bounds from global methane budget (part of GCP)
p, cov = scipy.optimize.curve_fit(
    fit_precursors,
    emis_obs,
    input_obs["SF6"],
)

parameters = {}

parameters["best_fit"] = {
    "base": p[0],
}

# these are the feedback values per ppb / per Mt that go into FaIR
print(parameters["best_fit"])

conc_sf6 = np.zeros(273)
gas_boxes = 0
airborne_emissions = 0

for i in range(273):
    conc_sf6[i], gas_boxes, airborne_emissions = one_box(
        emis_obs[i],
        gas_boxes,
        airborne_emissions,
        burden_per_emission,
        parameters["best_fit"]["base"],
        1,#lifetime_scaling["best_fit"][i],
        partition_fraction,
        pre_industrial_concentration=0,
        timestep=1,
        natural_emissions_adjustment=0#natural_emissions_adjustment,
    )

if plots:
    fig, ax = pl.subplots(1, 1, figsize=(3.5, 3.5))

    ax.plot(
        np.arange(1930, 2023), conc_sf6[180:], color="0.5", label="Best fit"
    )
    ax.plot(
        np.arange(1930, 2023), input_obs["SF6"][180:], color="k", label="observations"
    )
    ax.set_ylabel("ppt")
    ax.set_xlim(1930, 2023)
    ax.legend(frameon=False)
    ax.set_title("SF$_6$ concentration")

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "sf6_calibrations.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "sf6_calibrations.pdf"
    )
    pl.close()

# # these are the feedback values that go into FaIR
# out = np.empty((1, 1))
# out[0, 0] = parameters["best_fit"]["base"]
#
# df = pd.DataFrame(
#     out,
#     columns=["base"],
#     index=["historical_best"],
# )
# os.makedirs(
#     f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
#     exist_ok=True
# )
# df.to_csv(
#     f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
#     "SF6_lifetime.csv"
# )
