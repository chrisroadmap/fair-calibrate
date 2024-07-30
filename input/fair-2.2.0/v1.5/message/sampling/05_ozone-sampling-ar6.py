#!/usr/bin/env python
# coding: utf-8

"""Ozone calibration"""

# this is also without EESC, which could be a little more problematic.

import copy
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

print("Doing ozone sampling...")

# now include temperature feedback
Tobs = pd.read_csv(
    "../../../../../data/forcing/IGCC_GMST_1850-2023.csv", index_col=0
).values

delta_gmst = [
    0,
    Tobs[65:76].mean(),
    Tobs[75:86].mean(),
    Tobs[85:96].mean(),
    Tobs[95:106].mean(),
    Tobs[105:116].mean(),
    Tobs[115:126].mean(),
    Tobs[125:136].mean(),
    Tobs[135:146].mean(),
    Tobs[145:156].mean(),
    Tobs[152:163].mean(),
    Tobs[155:166].mean(),
    Tobs[159:170].mean(),
    Tobs[167].mean(),  # we don't use this
    Tobs[168].mean(),
]
warming_pi_pd = Tobs[159:170].mean()

assert fair_v == __version__

good_models = [
    "BCC-ESM1",
    "CESM2(WACCM6)",
    "GFDL-ESM4",
    "GISS-E2-1-H",
    "MRI-ESM2-0",
    "OsloCTM3",
]
skeie_trop = pd.read_csv(
    "../../../../../data/forcing/skeie_ozone_trop.csv", index_col=0
)
skeie_trop = skeie_trop.loc[good_models]
skeie_trop.insert(0, 1850, 0)
skeie_trop.columns = pd.to_numeric(skeie_trop.columns)
skeie_trop.interpolate(axis=1, method="values", limit_area="inside", inplace=True)

skeie_strat = pd.read_csv(
    "../../../../../data/forcing/skeie_ozone_strat.csv", index_col=0
)
skeie_strat = skeie_strat.loc[good_models]
skeie_strat.insert(0, 1850, 0)
skeie_strat.columns = pd.to_numeric(skeie_strat.columns)
skeie_strat.interpolate(axis=1, method="values", limit_area="inside", inplace=True)

skeie_total = skeie_trop + skeie_strat

coupled_models = copy.deepcopy(good_models)
coupled_models.remove("OsloCTM3")

skeie_total.loc[coupled_models] = skeie_total.loc[coupled_models] - (-0.037) * np.array(
    delta_gmst
)
skeie_ssp245 = skeie_total.mean()
skeie_ssp245[1750] = -0.03
skeie_ssp245.sort_index(inplace=True)
skeie_ssp245 = skeie_ssp245 + 0.03
skeie_ssp245.drop([2014, 2017, 2020], inplace=True)
skeie_ssp245 = pd.concat(
    (
        skeie_ssp245,
        skeie_total.loc["OsloCTM3", 2014:]
        - skeie_total.loc["OsloCTM3", 2010]
        + skeie_ssp245[2010],
    )
)

f = interp1d(
    skeie_ssp245.index, skeie_ssp245, bounds_error=False, fill_value="extrapolate"
)
years = np.arange(1750, 2021)
o3total = f(years)

print("2014-1750 ozone ERF from Skeie:", o3total[264])
print("2019-1750 ozone ERF from Skeie:", o3total[269])
print("2014-1850 ozone ERF from Skeie:", o3total[264] - o3total[100])

# Sampling with updated emissions.
df_emis_message = pd.read_csv(
    "../../../../../data/emissions/message-baseline-2020.csv",
    index_col=0,
)

# and observed concentrations.
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2023.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

emitted_species = [
    "NOx",
    "VOC",
    "CO",
]

concentration_species = [
    "CH4",
    "N2O",
]

species_out = {}

years_in = df_emis_message.loc[:,'1750.5':].columns.to_numpy().astype(float)
for ispec, species in enumerate(emitted_species):
    raw_data = (
        df_emis_message.loc[
            (df_emis_message["Scenario"] == "baseline")
            & (df_emis_message["Variable"] == species)
            & (df_emis_message["Region"] == "World"),
            "1750.5":"2105.5",
        ]
        .values.squeeze()
    )
    interpolator = interp1d(years_in, raw_data)
    species_out[species] = interpolator(np.arange(1750.5, 2023))

for ispec, species in enumerate(concentration_species):
    raw_data = df_conc_obs[species].values
    interpolator = interp1d(np.arange(1750, 2024), raw_data)
    species_out[species] = interpolator(np.arange(1750.5, 2023))

species_df = pd.DataFrame(species_out, index=np.arange(1750.5, 2023))

delta_Cch4 = species_out["CH4"][264] - species_out["CH4"][0]
delta_Cn2o = species_out["N2O"][264] - species_out["N2O"][0]
delta_Eco = species_out["CO"][264] - species_out["CO"][0]
delta_Evoc = species_out["VOC"][264] - species_out["VOC"][0]
delta_Enox = species_out["NOx"][264] - species_out["NOx"][0]

# best estimate radiative efficienices from 2014 - 1850 from AR6 here
radeff_ch4 = 0.14 / delta_Cch4
radeff_n2o = 0.03 / delta_Cn2o
radeff_co = 0.067 / delta_Eco  # stevenson CMIP5 scaled to CO + VOC total
radeff_voc = 0.043 / delta_Evoc  # stevenson CMIP5 scaled to CO + VOC total
radeff_nox = 0.20 / delta_Enox


fac_cmip6_skeie = (
    radeff_ch4 * delta_Cch4
    + radeff_n2o * delta_Cn2o
    + radeff_co * delta_Eco
    + radeff_voc * delta_Evoc
    + radeff_nox * delta_Enox
) / (o3total[264] - o3total[0])
ts = np.vstack(
    (
        species_out["CH4"],
        species_out["N2O"],
        species_out["CO"],
        species_out["VOC"],
        species_out["NOx"],
    )
).T


def fit_precursors(x, rch4, rn2o, rco, rvoc, rnox):
    return (
        rch4 * x[0] + rn2o * x[1] + rco * x[2] + rvoc * x[3] + rnox * x[4]
    )


p, cov = curve_fit(
    fit_precursors,
    ts[:270, :].T - ts[0:1, :].T,
    o3total[:270] - o3total[0],
    bounds=(  # very likely range from Thornhill - maybe could be wider?
        (
            0.09 / delta_Cch4 / fac_cmip6_skeie,
            0.01 / delta_Cn2o / fac_cmip6_skeie,
            0.010 / delta_Eco / fac_cmip6_skeie,
            0 / delta_Evoc / fac_cmip6_skeie,
            0.09 / delta_Enox / fac_cmip6_skeie,
        ),
        (
            0.19 / delta_Cch4 / fac_cmip6_skeie,
            0.05 / delta_Cn2o / fac_cmip6_skeie,
            0.124 / delta_Eco / fac_cmip6_skeie,
            0.086 / delta_Evoc / fac_cmip6_skeie,
            0.31 / delta_Enox / fac_cmip6_skeie,
        ),
    ),
)

forcing = (
    p[0] * (species_out["CH4"] - species_out["CH4"][0])
    + p[1] * (species_out["N2O"] - species_out["N2O"][0])
    + p[2] * (species_out["CO"] - species_out["CO"][0])
    + p[3] * (species_out["VOC"] - species_out["VOC"][0])
    + p[4] * (species_out["NOx"] - species_out["NOx"][0])
)

if plots:
    pl.figure(figsize=(9 / 2.54, 9 / 2.54))
    pl.plot(np.arange(1750.5, 2023), forcing, label="best estimate fit", color="0.5")
    pl.plot(np.arange(1750.5, 2021), o3total, label="Skeie et al. 2020 mean", color="k")
    pl.legend()
    pl.title("Ozone forcing (no feedbacks)")
    pl.ylabel("W m$^{-2}$")
    pl.xlim(1750, 2023)
    pl.tight_layout()
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}", exist_ok=True
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ozone_calibration.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ozone_calibration.pdf"
    )
    pl.close()

print(p)  # these coefficients we export to the ERF time series
# print(radeff_ch4, radeff_n2o, radeff_ods, radeff_co, radeff_voc, radeff_nox)

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

scalings = scipy.stats.norm.rvs(
    loc=np.array(p),
    scale=np.array(
        [
            0.05 / delta_Cch4 / fac_cmip6_skeie,
            0.02 / delta_Cn2o / fac_cmip6_skeie,
            0.057 / delta_Eco / fac_cmip6_skeie,
            0.043 / delta_Evoc / fac_cmip6_skeie,
            0.11 / delta_Enox / fac_cmip6_skeie,
        ]
    )
    # scale=np.array([0.000062, 0.000471, 0.000113, 0.000131, 0.000328, 0.000983])
    / NINETY_TO_ONESIGMA,
    size=(samples, 5),
    random_state=52,
)

df = pd.DataFrame(
    scalings,
    columns=[
        "CH4",
        "N2O",
        "CO",
        "VOC",
        "NOx",
    ],
)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv",
    index=False,
)
