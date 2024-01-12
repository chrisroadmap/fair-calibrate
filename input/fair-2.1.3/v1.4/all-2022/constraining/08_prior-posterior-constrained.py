#!/usr/bin/env python
# coding: utf-8

"""Plot priors and posteriors of all parameters, if plots requested."""

import os
import sys

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__

# if we're not plotting, don't even start
load_dotenv()
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")

if not plots:
    sys.exit()

# pl.switch_backend("agg")

pl.style.use("../../../../../defaults.mplstyle")

# override font size
pl.rcParams["font.size"] = 6

print("Making parameters plot...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

assert fair_v == __version__


step1 = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv",
    dtype="int",
)
accept_step1 = np.zeros(samples, dtype=bool)
accept_step1[step1] = True

step2 = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    dtype="int",
)
accept_step2 = np.zeros(samples, dtype=bool)
accept_step2[step2] = True

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

df_cc = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "carbon_cycle.csv"
)
df_cr = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "climate_response_ebm3.csv"
)
df_aci = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_cloud.csv"
)
df_ari = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_radiation.csv"
)
df_ozone = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/" "ozone.csv"
)
df_scaling = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "forcing_scaling.csv"
)
df_1750co2 = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "co2_concentration_1750.csv"
)

negative_only_beta = df_aci.loc[:, "beta"].values
negative_only_beta[negative_only_beta >= 0] = np.nan

prior = np.vstack(
    (
        df_1750co2.loc[:, "co2_concentration"].values,
        df_cc.loc[:, "rA"].values,
        df_cc.loc[:, "rU"].values,
        df_cc.loc[:, "rT"].values,
        df_cc.loc[:, "r0"].values,
        df_ari.loc[:, "BC"].values,
        df_ari.loc[:, "OC"].values,
        df_ari.loc[:, "Sulfur"].values,
        df_ari.loc[:, "NH3"].values,
        df_ari.loc[:, "NOx"].values,
        df_ari.loc[:, "CH4"].values,
        df_ari.loc[:, "N2O"].values,
        df_ari.loc[:, "VOC"].values,
        df_ari.loc[:, "Equivalent effective stratospheric chlorine"].values,
        np.log(df_aci.loc[:, "shape_so2"].values),
        np.log(df_aci.loc[:, "shape_bc"].values),
        np.log(df_aci.loc[:, "shape_oc"].values),
        # df_aci.loc[:, "beta"].values,
        np.log(-negative_only_beta),
        df_cr.loc[:, "F_4xCO2"].values,
        df_cr.loc[:, "c1"].values,
        df_cr.loc[:, "c2"].values,
        df_cr.loc[:, "c3"].values,
        df_cr.loc[:, "kappa1"].values,
        df_cr.loc[:, "kappa2"].values,
        df_cr.loc[:, "kappa3"].values,
        df_cr.loc[:, "epsilon"].values,
        df_cr.loc[:, "gamma"].values,
        df_cr.loc[:, "sigma_eta"].values,
        df_cr.loc[:, "sigma_xi"].values,
        df_ozone.loc[:, "CH4"].values,
        df_ozone.loc[:, "N2O"].values,
        df_ozone.loc[:, "Equivalent effective stratospheric chlorine"].values,
        df_ozone.loc[:, "VOC"].values,
        df_ozone.loc[:, "CO"].values,
        df_ozone.loc[:, "NOx"].values,
        df_scaling.loc[:, "CO2"].values,
        df_scaling.loc[:, "CH4"].values,
        df_scaling.loc[:, "N2O"].values,
        df_scaling.loc[:, "minorGHG"].values,
        df_scaling.loc[:, "Stratospheric water vapour"].values,
        df_scaling.loc[:, "Light absorbing particles on snow and ice"].values,
        df_scaling.loc[:, "Land use"].values,
        df_scaling.loc[:, "Volcanic"].values,
        df_scaling.loc[:, "solar_trend"].values,
        df_scaling.loc[:, "solar_amplitude"].values,
    ),
)

titles = [
    r"CO$_2$ conc. 1750",
    "iirf_airborne",
    "iirf_uptake",
    "iirf_temperature",
    "iirf_1750",
    "ARI BC",
    "ARI OC",
    r"ARI SO$_2$",
    r"ARI NH$_3$",
    "ARI NOx",
    "ARI CH$_4$",
    "ARI N$_2$O",
    "ARI VOC",
    "ARI EESC",
    "log(ACI shape BC)",
    "log(ACI shape OC)",
    "log(ACI shape SO2)",
    "-log(ACI scale)",
    r"F_4$\times$CO$_2$",
    "c1",
    "c2",
    "c3",
    "kappa1",
    "kappa2",
    "kappa3",
    "epsilon",
    "gamma",
    "sigma_eta",
    "sigma_xi",
    "o3_CH4",
    "o3_N2O",
    "o3_EESC",
    "o3_VOC",
    "o3_CO",
    "o3_VOC",
    "scale_CO2",
    "scale_CH4",
    "scale_N2O",
    "scale_minorGHG",
    "scale_stratH2O",
    "scale_LAPSI",
    "scale_landuse",
    "scale_volcanic",
    "solar_trend",
    "solar_amplitude",
]

fig, ax = pl.subplots(9, 5, figsize=(18 / 2.54, 18 / 2.54))
for isp in range(45):
    row = isp // 5
    col = isp % 5

    min = np.nanmin(prior[isp, :])
    max = np.nanmax(prior[isp, :])
    bins = np.linspace(min, max, 41)
    ax[row, col].hist(
        prior[isp, :], density=True, alpha=0.5, bins=bins, color=colors["prior"]
    )
    ax[row, col].hist(
        prior[isp, accept_step2],
        density=True,
        alpha=0.5,
        bins=bins,
        color=colors["post2"],
    )
    ax[row, col].set_xlim(min, max)
    ax[row, col].set_yticklabels([])
    ax[row, col].set_title(titles[isp])

pl.tight_layout()
pl.savefig(
    f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/" "parameters.png"
)
pl.savefig(
    f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/" "parameters.pdf"
)
pl.close()
