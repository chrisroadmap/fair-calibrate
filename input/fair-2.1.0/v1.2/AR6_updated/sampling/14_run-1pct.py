#!/usr/bin/env python
# coding: utf-8

"""Run concentration driven FaIR 2.1 for SSP245"""

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
from parallel_1pct import run_fair
from utils import _parallel_process

if __name__ == "__main__":
    print("Running 1pct scenarios (could take a while)...")
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

    df_cc = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "carbon_cycle.csv"
    )
    df_cr = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "climate_response_ebm3.csv"
    )
    df_scaling = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "forcing_scaling.csv"
    )

    seedgen = 1355763
    seedstep = 399

    # we only care about temperature and airborne fraction in years 70 and 140
    # as well as the temperature at 1000 GtC
    temp_2x4x_out = np.ones((2, samples)) * np.nan
    temp_1000_out = np.ones((samples)) * np.nan
    af_out = np.ones((2, samples)) * np.nan

    config = []
    for ibatch, batch_start in enumerate(range(0, samples, batch_size)):
        config.append({})
        batch_end = min(batch_start + batch_size, samples)
        config[ibatch]["batch_start"] = batch_start
        config[ibatch]["batch_end"] = batch_start + batch_size
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
        config[ibatch]["scaling_CO2"] = df_scaling.loc[
            batch_start : batch_end - 1, "CO2"
        ].values.squeeze()
        config[ibatch]["scaling_CH4"] = df_scaling.loc[
            batch_start : batch_end - 1, "CH4"
        ].values.squeeze()
        config[ibatch]["scaling_N2O"] = df_scaling.loc[
            batch_start : batch_end - 1, "N2O"
        ].values.squeeze()

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
        batch_end = min(batch_start + batch_size, samples)
        temp_2x4x_out[:, batch_start:batch_end] = res[ibatch][0]
        af_out[:, batch_start:batch_end] = res[ibatch][1]
        temp_1000_out[batch_start:batch_end] = res[ibatch][2]

    os.makedirs(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/",
        exist_ok=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "temperature_1pctCO2_y70_y140.npy",
        temp_2x4x_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "airborne_fraction_1pctCO2_y70_y140.npy",
        af_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "temperature_1pctCO2_1000GtC.npy",
        temp_1000_out,
        allow_pickle=True,
    )
