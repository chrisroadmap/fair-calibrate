#!/usr/bin/env python
# coding: utf-8

"""Ozone calibration"""

# concentrations are midyear rather than endyear so are six months out, but it won't
# be a biggie.

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
    "../../../../../data/forcing/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.rebased_1850-1900.csv", index_col=0
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

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)
df_conc = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2022.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc.loc[year, :] = np.nan
df_conc.sort_index(inplace=True)
df_conc.interpolate(inplace=True)

emitted_species = [
    "NOx",
    "VOC",
    "CO",
]

concentration_species = [
    "CH4",
    "N2O",
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
]


hc_species = [
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
]

name_conv = {specie: specie for specie in emitted_species}
name_conv["VOC"] = "NMVOC"

species_out = {}
for ispec, species in enumerate(emitted_species):
    species_out[species] = df_emis[name_conv[species]].values

for ispec, species in enumerate(concentration_species):
    species_out[species] = df_conc[species].values[:273]

species_df = pd.DataFrame(species_out, index=range(1750, 2023))


def calculate_eesc(
    concentration,
    fractional_release,
    fractional_release_cfc11,
    cl_atoms,
    br_atoms,
    br_cl_ratio=45,
):
    # EESC is in terms of CFC11-eq
    eesc_out = (
        cl_atoms * (concentration) * fractional_release / fractional_release_cfc11
        + br_cl_ratio
        * br_atoms
        * (concentration)
        * fractional_release
        / fractional_release_cfc11
    ) * fractional_release_cfc11
    return eesc_out


fractional_release = {
    "CFC-11": 0.47,
    "CFC-12": 0.23,
    "CFC-113": 0.29,
    "CFC-114": 0.12,
    "CFC-115": 0.04,
    "HCFC-22": 0.13,
    "HCFC-141b": 0.34,
    "HCFC-142b": 0.17,
    "CCl4": 0.56,
    "CHCl3": 0,
    "CH2Cl2": 0,
    "CH3Cl": 0.44,
    "CH3CCl3": 0.67,
    "CH3Br": 0.6,
    "Halon-1211": 0.62,
    "Halon-1301": 0.28,
    "Halon-2402": 0.65,
}

cl_atoms = {
    "CFC-11": 3,
    "CFC-12": 2,
    "CFC-113": 3,
    "CFC-114": 2,
    "CFC-115": 1,
    "HCFC-22": 1,
    "HCFC-141b": 2,
    "HCFC-142b": 1,
    "CCl4": 4,
    "CHCl3": 3,
    "CH2Cl2": 2,
    "CH3Cl": 1,
    "CH3CCl3": 3,
    "CH3Br": 0,
    "Halon-1211": 1,
    "Halon-1301": 0,
    "Halon-2402": 0,
}

br_atoms = {
    "CFC-11": 0,
    "CFC-12": 0,
    "CFC-113": 0,
    "CFC-114": 0,
    "CFC-115": 0,
    "HCFC-22": 0,
    "HCFC-141b": 0,
    "HCFC-142b": 0,
    "CCl4": 0,
    "CHCl3": 0,
    "CH2Cl2": 0,
    "CH3Cl": 0,
    "CH3CCl3": 0,
    "CH3Br": 1,
    "Halon-1211": 1,
    "Halon-1301": 1,
    "Halon-2402": 2,
}

hc_eesc = {}
total_eesc = 0
for species in hc_species:
    hc_eesc[species] = calculate_eesc(
        species_df.loc[:, species],
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc = total_eesc + hc_eesc[species].values

delta_Cch4 = species_out["CH4"][264] - species_out["CH4"][0]
delta_Cn2o = species_out["N2O"][264] - species_out["N2O"][0]
delta_Cods = total_eesc[264] - total_eesc[0]
delta_Eco = species_out["CO"][264] - species_out["CO"][0]
delta_Evoc = species_out["VOC"][264] - species_out["VOC"][0]
delta_Enox = species_out["NOx"][264] - species_out["NOx"][0]

# best estimate radiative efficienices from 2014 - 1850 from AR6 here
radeff_ch4 = 0.14 / delta_Cch4
radeff_n2o = 0.03 / delta_Cn2o
radeff_ods = -0.11 / delta_Cods
radeff_co = 0.067 / delta_Eco  # stevenson CMIP5 scaled to CO + VOC total
radeff_voc = 0.043 / delta_Evoc  # stevenson CMIP5 scaled to CO + VOC total
radeff_nox = 0.20 / delta_Enox


fac_cmip6_skeie = (
    radeff_ch4 * delta_Cch4
    + radeff_n2o * delta_Cn2o
    + radeff_ods * delta_Cods
    + radeff_co * delta_Eco
    + radeff_voc * delta_Evoc
    + radeff_nox * delta_Enox
) / (o3total[264] - o3total[0])
ts = np.vstack(
    (
        species_out["CH4"],
        species_out["N2O"],
        total_eesc,
        species_out["CO"],
        species_out["VOC"],
        species_out["NOx"],
    )
).T


def fit_precursors(x, rch4, rn2o, rods, rco, rvoc, rnox):
    return (
        rch4 * x[0] + rn2o * x[1] + rods * x[2] + rco * x[3] + rvoc * x[4] + rnox * x[5]
    )


p, cov = curve_fit(
    fit_precursors,
    ts[:270, :].T - ts[0:1, :].T,
    o3total[:270] - o3total[0],
    bounds=(  # very likely range from Thornhill - maybe could be wider?
        (
            0.09 / delta_Cch4 / fac_cmip6_skeie,
            0.01 / delta_Cn2o / fac_cmip6_skeie,
            -0.21 / delta_Cods / fac_cmip6_skeie,
            0.010 / delta_Eco / fac_cmip6_skeie,
            0 / delta_Evoc / fac_cmip6_skeie,
            0.09 / delta_Enox / fac_cmip6_skeie,
        ),
        (
            0.19 / delta_Cch4 / fac_cmip6_skeie,
            0.05 / delta_Cn2o / fac_cmip6_skeie,
            -0.01 / delta_Cods / fac_cmip6_skeie,
            0.124 / delta_Eco / fac_cmip6_skeie,
            0.086 / delta_Evoc / fac_cmip6_skeie,
            0.31 / delta_Enox / fac_cmip6_skeie,
        ),
    ),
)

forcing = (
    p[0] * (species_out["CH4"] - species_out["CH4"][0])
    + p[1] * (species_out["N2O"] - species_out["N2O"][0])
    + p[2] * (total_eesc - total_eesc[0])
    + p[3] * (species_out["CO"] - species_out["CO"][0])
    + p[4] * (species_out["VOC"] - species_out["VOC"][0])
    + p[5] * (species_out["NOx"] - species_out["NOx"][0])
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
            0.10 / delta_Cods / fac_cmip6_skeie,
            0.057 / delta_Eco / fac_cmip6_skeie,
            0.043 / delta_Evoc / fac_cmip6_skeie,
            0.11 / delta_Enox / fac_cmip6_skeie,
        ]
    )
    # scale=np.array([0.000062, 0.000471, 0.000113, 0.000131, 0.000328, 0.000983])
    / NINETY_TO_ONESIGMA,
    size=(samples, 6),
    random_state=52,
)

# 90% ranges - for paper
print(
    np.array(
        [
            0.05 / delta_Cch4 / fac_cmip6_skeie,
            0.02 / delta_Cn2o / fac_cmip6_skeie,
            0.10 / delta_Cods / fac_cmip6_skeie,
            0.057 / delta_Eco / fac_cmip6_skeie,
            0.043 / delta_Evoc / fac_cmip6_skeie,
            0.11 / delta_Enox / fac_cmip6_skeie,
        ]
    )
)

df = pd.DataFrame(
    scalings,
    columns=[
        "CH4",
        "N2O",
        "Equivalent effective stratospheric chlorine",
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
