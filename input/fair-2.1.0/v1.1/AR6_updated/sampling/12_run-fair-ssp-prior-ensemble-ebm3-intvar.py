#!/usr/bin/env python
# coding: utf-8

"""AR6 calibration of FaIR 2.1"""
#
# - use logsum aerosol indirect forcing relationship.
# - aerosol direct is from AR6.
# - three layer model for climate response.
# - include overlap of the major GHGs.
# - prognostic equation for land use related forcing (e.g. from FaIR 1.6).
# - ozone relationship from FaIR 1.6 used in AR6.
# - interactive methane lifetime
# - internal variability (here included)
#
# We have to do this slightly differently to the examples so far. 1.5 million ensemble
# members is going to take up too much memory, so we run in batches of 1000,
# initialising a new FaIR instance for each batch, and saving the output as we go.


import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from parallel import run_fair
from utils import _parallel_process

if __name__ == "__main__":
    print("Running the priors (could take a while)...")
    load_dotenv()

    cal_v = os.getenv("CALIBRATION_VERSION")
    fair_v = os.getenv("FAIR_VERSION")
    constraint_set = os.getenv("CONSTRAINT_SET")
    samples = int(os.getenv("PRIOR_SAMPLES"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    WORKERS = int(os.getenv("WORKERS"))

    # number of processors
    WORKERS = min(multiprocessing.cpu_count(), WORKERS)

    assert fair_v == __version__

    df_solar = pd.read_csv(
        "../../../../../data/forcing/solar_erf_timebounds.csv", index_col="year"
    )
    df_volcanic = pd.read_csv(
        "../../../../../data/forcing/volcanic_ERF_monthly_-950001-201912.csv"
    )

    volcanic_forcing = np.zeros(352)
    for i, year in enumerate(np.arange(1750, 2021)):
        volcanic_forcing[i] = np.mean(
            df_volcanic.loc[
                ((year - 1) <= df_volcanic["year"]) & (df_volcanic["year"] < year)
            ].erf
        )
    volcanic_forcing[271:281] = np.linspace(1, 0, 10) * volcanic_forcing[270]

    solar_forcing = df_solar["erf"].loc[1750:2101].values

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
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "ozone.csv"
    )
    df_scaling = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "forcing_scaling.csv"
    )
    df_1750co2 = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "co2_concentration_1750.csv"
    )
    df_methane = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
        "CH4_lifetime.csv",
        index_col=0,
    )

    seedgen = 1355763
    seedstep = 399

    # for all except temperature, the full time series is not important so we can save
    # a bit of space
    temp_out = np.ones((252, samples)) * np.nan
    ohc_out = np.ones((samples)) * np.nan
    fari_out = np.ones((samples)) * np.nan
    faci_out = np.ones((samples)) * np.nan
    co2_out = np.ones((samples)) * np.nan
    fo3_out = np.ones((samples)) * np.nan
    ecs = np.ones(samples) * np.nan
    tcr = np.ones(samples) * np.nan

    calibrated_f4co2_mean = df_cr["F_4xCO2"].mean()

    config = []
    for ibatch, batch_start in enumerate(range(0, samples, batch_size)):
        config.append({})
        batch_end = batch_start + batch_size
        config[ibatch]["batch_start"] = batch_start
        config[ibatch]["batch_end"] = batch_start + batch_size
        config[ibatch]["volcanic_forcing"] = volcanic_forcing
        config[ibatch]["solar_forcing"] = solar_forcing
        config[ibatch]["scaling_Volcanic"] = df_scaling.loc[
            batch_start : batch_end - 1, "Volcanic"
        ].values.squeeze()
        config[ibatch]["scaling_solar_trend"] = df_scaling.loc[
            batch_start : batch_end - 1, "solar_trend"
        ].values.squeeze()
        config[ibatch]["scaling_solar_amplitude"] = df_scaling.loc[
            batch_start : batch_end - 1, "solar_amplitude"
        ].values.squeeze()
        config[ibatch]["c1"] = df_cr.loc[batch_start : batch_end - 1, "c1"].values
        config[ibatch]["c2"] = df_cr.loc[batch_start : batch_end - 1, "c2"].values
        config[ibatch]["c3"] = df_cr.loc[batch_start : batch_end - 1, "c3"].values
        config[ibatch]["kappa1"] = df_cr.loc[
            batch_start : batch_end - 1, "kappa1"
        ].values
        config[ibatch]["kappa2"] = df_cr.loc[
            batch_start : batch_end - 1, "kappa2"
        ].values
        config[ibatch]["kappa3"] = df_cr.loc[
            batch_start : batch_end - 1, "kappa3"
        ].values
        config[ibatch]["epsilon"] = df_cr.loc[
            batch_start : batch_end - 1, "epsilon"
        ].values
        config[ibatch]["gamma"] = df_cr.loc[batch_start : batch_end - 1, "gamma"].values
        config[ibatch]["sigma_eta"] = df_cr.loc[
            batch_start : batch_end - 1, "sigma_eta"
        ].values
        config[ibatch]["sigma_xi"] = df_cr.loc[
            batch_start : batch_end - 1, "sigma_xi"
        ].values
        config[ibatch]["seed"] = np.arange(
            seedgen + batch_start * seedstep,
            seedgen + batch_end * seedstep,
            seedstep,
            dtype=int,
        )
        config[ibatch]["forcing_4co2"] = df_cr.loc[
            batch_start : batch_end - 1, "F_4xCO2"
        ]
        config[ibatch]["iirf_0"] = df_cc.loc[
            batch_start : batch_end - 1, "r0"
        ].values.squeeze()
        config[ibatch]["iirf_airborne"] = df_cc.loc[
            batch_start : batch_end - 1, "rA"
        ].values.squeeze()
        config[ibatch]["iirf_uptake"] = df_cc.loc[
            batch_start : batch_end - 1, "rU"
        ].values.squeeze()
        config[ibatch]["iirf_temperature"] = df_cc.loc[
            batch_start : batch_end - 1, "rT"
        ].values.squeeze()
        config[ibatch]["beta"] = df_aci.loc[
            batch_start : batch_end - 1, "beta"
        ].values.squeeze()
        config[ibatch]["shape_so2"] = df_aci.loc[
            batch_start : batch_end - 1, "shape_so2"
        ].values.squeeze()
        config[ibatch]["shape_bc"] = df_aci.loc[
            batch_start : batch_end - 1, "shape_bc"
        ].values.squeeze()
        config[ibatch]["shape_oc"] = df_aci.loc[
            batch_start : batch_end - 1, "shape_oc"
        ].values.squeeze()
        config[ibatch]["scaling_CO2"] = df_scaling.loc[
            batch_start : batch_end - 1, "CO2"
        ].values.squeeze()
        config[ibatch]["scaling_CH4"] = df_scaling.loc[
            batch_start : batch_end - 1, "CH4"
        ].values.squeeze()
        config[ibatch]["scaling_N2O"] = df_scaling.loc[
            batch_start : batch_end - 1, "N2O"
        ].values.squeeze()
        config[ibatch]["scaling_minorGHG"] = df_scaling.loc[
            batch_start : batch_end - 1, "minorGHG"
        ].values.squeeze()
        config[ibatch]["scaling_stwv"] = df_scaling.loc[
            batch_start : batch_end - 1, "Stratospheric water vapour"
        ].values.squeeze()
        config[ibatch]["scaling_contrails"] = df_scaling.loc[
            batch_start : batch_end - 1, "Contrails"
        ].values.squeeze()
        config[ibatch]["scaling_lapsi"] = df_scaling.loc[
            batch_start : batch_end - 1, "Light absorbing particles on snow and ice"
        ].values.squeeze()
        config[ibatch]["scaling_landuse"] = df_scaling.loc[
            batch_start : batch_end - 1, "Land use"
        ].values.squeeze()
        config[ibatch]["ari_BC"] = df_ari.loc[batch_start : batch_end - 1, "BC"]
        config[ibatch]["ari_CH4"] = df_ari.loc[batch_start : batch_end - 1, "CH4"]
        config[ibatch]["ari_CO"] = df_ari.loc[batch_start : batch_end - 1, "CO"]
        config[ibatch]["ari_N2O"] = df_ari.loc[batch_start : batch_end - 1, "N2O"]
        config[ibatch]["ari_NH3"] = df_ari.loc[batch_start : batch_end - 1, "NH3"]
        config[ibatch]["ari_NOx"] = df_ari.loc[batch_start : batch_end - 1, "NOx"]
        config[ibatch]["ari_OC"] = df_ari.loc[batch_start : batch_end - 1, "OC"]
        config[ibatch]["ari_Sulfur"] = df_ari.loc[batch_start : batch_end - 1, "Sulfur"]
        config[ibatch]["ari_VOC"] = df_ari.loc[batch_start : batch_end - 1, "VOC"]
        config[ibatch]["ari_EESC"] = df_ari.loc[
            batch_start : batch_end - 1, "Equivalent effective stratospheric chlorine"
        ]
        config[ibatch]["ozone_CH4"] = df_ozone.loc[batch_start : batch_end - 1, "CH4"]
        config[ibatch]["ozone_N2O"] = df_ozone.loc[batch_start : batch_end - 1, "N2O"]
        config[ibatch]["ozone_NOx"] = df_ozone.loc[batch_start : batch_end - 1, "NOx"]
        config[ibatch]["ozone_VOC"] = df_ozone.loc[batch_start : batch_end - 1, "VOC"]
        config[ibatch]["ozone_CO"] = df_ozone.loc[batch_start : batch_end - 1, "CO"]
        config[ibatch]["ozone_EESC"] = df_ozone.loc[
            batch_start : batch_end - 1, "Equivalent effective stratospheric chlorine"
        ]
        config[ibatch]["CO2_1750"] = df_1750co2.loc[
            batch_start : batch_end - 1, "co2_concentration"
        ].values.squeeze()
        config[ibatch]["ch4_base"] = df_methane.loc["historical_best", "base"]
        config[ibatch]["ch4_CH4"] = df_methane.loc["historical_best", "CH4"]
        config[ibatch]["ch4_NOx"] = df_methane.loc["historical_best", "NOx"]
        config[ibatch]["ch4_VOC"] = df_methane.loc["historical_best", "VOC"]
        config[ibatch]["ch4_EESC"] = df_methane.loc["historical_best", "HC"]
        config[ibatch]["ch4_N2O"] = df_methane.loc["historical_best", "N2O"]
        config[ibatch]["ch4_temp"] = df_methane.loc["historical_best", "temp"]

    parallel_process_kwargs = dict(
        func=run_fair,
        configuration=config,
        config_are_kwargs=False,
    )

    with ProcessPoolExecutor(WORKERS) as pool:
        res = _parallel_process(
            **parallel_process_kwargs,
            pool=pool,
        )

    for ibatch, batch_start in enumerate(range(0, samples, batch_size)):
        batch_end = batch_start + batch_size
        temp_out[:, batch_start:batch_end] = res[ibatch][0]
        ohc_out[batch_start:batch_end] = res[ibatch][1]
        co2_out[batch_start:batch_end] = res[ibatch][2]
        fari_out[batch_start:batch_end] = res[ibatch][3]
        faci_out[batch_start:batch_end] = res[ibatch][4]
        ecs[batch_start:batch_end] = res[ibatch][5]
        tcr[batch_start:batch_end] = res[ibatch][6]

    os.makedirs(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/",
        exist_ok=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "temperature_1850-2101.npy",
        temp_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "ocean_heat_content_2018_minus_1971.npy",
        ohc_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "concentration_co2_2014.npy",
        co2_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "forcing_ari_2005-2014_mean.npy",
        fari_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "forcing_aci_2005-2014_mean.npy",
        faci_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "ecs.npy",
        ecs,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "tcr.npy",
        tcr,
        allow_pickle=True,
    )
