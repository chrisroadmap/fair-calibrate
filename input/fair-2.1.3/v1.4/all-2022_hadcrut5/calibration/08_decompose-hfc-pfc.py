#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
import pooch
from dotenv import load_dotenv

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

rcmip_file = pooch.retrieve(
    url="https://zenodo.org/records/4589756/files/"
    "rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
    path=datadir,
    progressbar=progress,
)

gwp100ar6 = {
    "Emissions|F-Gases|HFC|HFC23": 14600,
    "Emissions|F-Gases|HFC|HFC32": 771,
    "Emissions|F-Gases|HFC|HFC125": 3740,
    "Emissions|F-Gases|HFC|HFC134a": 1530,
    "Emissions|F-Gases|HFC|HFC143a": 5810,
    "Emissions|F-Gases|HFC|HFC152a": 164,
    "Emissions|F-Gases|HFC|HFC227ea": 3600,
    "Emissions|F-Gases|HFC|HFC236fa": 8690,
    "Emissions|F-Gases|HFC|HFC245fa": 962,
    "Emissions|F-Gases|HFC|HFC365mfc": 914,
    "Emissions|F-Gases|HFC|HFC4310mee": 1600,
    "Emissions|F-Gases|PFC|CF4": 7380,
    "Emissions|F-Gases|PFC|C2F6": 12400,
    "Emissions|F-Gases|PFC|C3F8": 9290,
    "Emissions|F-Gases|PFC|cC4F8": 10200,
    "Emissions|F-Gases|PFC|C4F10": 10000,
    "Emissions|F-Gases|PFC|C5F12": 9220,
    "Emissions|F-Gases|PFC|C6F14": 8620,
    "Emissions|F-Gases|PFC|C7F16": 8410,
    "Emissions|F-Gases|PFC|C8F18": 8260,
}

rcmip_df = pd.read_csv(rcmip_file)
gases = rcmip_df.loc[
    (rcmip_df["Region"] == "World")
    & (rcmip_df["Scenario"] == "ssp245")
    & (
        (rcmip_df["Variable"].str.contains(r"\|HFC\|"))
        | (rcmip_df["Variable"].str.contains(r"\|PFC\|"))
    ),
    "1750":"2030",
].interpolate(axis=1)
gases_cols = rcmip_df.loc[
    (rcmip_df["Region"] == "World")
    & (rcmip_df["Scenario"] == "ssp245")
    & (
        (rcmip_df["Variable"].str.contains(r"\|HFC\|"))
        | (rcmip_df["Variable"].str.contains(r"\|PFC\|"))
    ),
    "Variable",
]
gases.drop(
    columns=["2024", "2025", "2026", "2027", "2028", "2029", "2030"], inplace=True
)

# Some of the emissions in RCMIP are below zero - do not allow this
gases[gases < 0] = 0
gases.index = gases_cols

# calculate CO2eq emissions for each gas
gases_gwp100ar6 = gases.multiply(pd.Series(gwp100ar6), axis=0)

# AR6 GWP100 as a one-row time series from RCMIP
hfc_rcmip_total = gases_gwp100ar6.loc[
    gases_gwp100ar6.index.str.contains(r"\|HFC\|"), :
].sum()
pfc_rcmip_total = gases_gwp100ar6.loc[
    gases_gwp100ar6.index.str.contains(r"\|PFC\|"), :
].sum()

# Compare to PRIMAP
primap_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "primap-histtp-2.5_1750-2022.csv",
    index_col=0,
)

# now scale the emissions from RCMIP to PRIMAP
scale_hfc_total = primap_df.loc["HFCs", "1750":"2022"].values / hfc_rcmip_total[:-1]
scale_hfc_total = scale_hfc_total.fillna(0)

scale_pfc_total = primap_df.loc["PFCs", "1750":"2022"].values / pfc_rcmip_total[:-1]
scale_pfc_total = scale_pfc_total.fillna(0)

hfcs_decomposed = (
    gases.loc[gases.index.str.contains(r"\|HFC\|"), "1750":"2022"] * scale_hfc_total
)
pfcs_decomposed = (
    gases.loc[gases.index.str.contains(r"\|PFC\|"), "1750":"2022"] * scale_pfc_total
)

df_out = pd.concat((hfcs_decomposed, pfcs_decomposed))

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions",
    exist_ok=True,
)
df_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "hfcs_pfcs_split_1750-2022.csv"
)
