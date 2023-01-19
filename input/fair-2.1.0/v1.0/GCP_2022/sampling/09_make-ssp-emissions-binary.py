#!/usr/bin/env python
# coding: utf-8

"""Make harmonized GCP CO2 emissions to SSPs then emissions binary file."""
#
# SSPs from RCMIP

import datetime
import os
import warnings

import aneris.convenience
import numpy as np
import pandas as pd
import pooch
import scmdata
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.interface import fill
from fair.io import read_properties
from tqdm.auto import tqdm

load_dotenv()


cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

print("Making SSP emissions binary...")

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

species, properties = read_properties()

f = FAIR(ch4_method="thornhill2021")
f.define_time(1750, 2500, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()
f.fill_from_rcmip()

# Do the harmonization
variables = [
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO2|AFOLU",
]

rcmip_variables = [
    "Emissions|CO2|MAGICC Fossil and Industrial",
    "Emissions|CO2|MAGICC AFOLU",
]


times = []
years = range(1750, 2023)
for year in years:
    times.append(datetime.datetime(year, 1, 1))
    # they are really midyears, but we just want this to work

times_future = []
years_future = range(2021, 2501)
for year in years_future:
    times_future.append(datetime.datetime(year, 1, 1))

history = (
    scmdata.ScmRun(
        "../../../../../data/emissions/gcp_iamc_format.csv", lowercase_cols=True
    )
    .filter(region="World", variable=variables)
    .interpolate(target_times=times)
    .timeseries(time_axis="year")
)
# I don't like scmdata's default conversion, so hack
# history["units"] = "Gt CO2/yr"
history = history * 44.009 / 12.011
arrays = []
for idx in range(0, len(history.index)):
    arrays.append(list(history.index[idx]))
    arrays[-1][3] = "GtCO2/yr"

new_index = pd.MultiIndex.from_tuples(
    list(zip(*list(map(list, zip(*arrays))))), names=history.index.names
)
history.index = new_index


rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)
future = (
    scmdata.ScmRun(rcmip_emissions_file, lowercase_cols=True)
    .filter(scenario=scenarios, variable=rcmip_variables, region="World")
    .interpolate(times_future)
    .timeseries(time_axis="year")
)
future = future / 1000

arrays = []
for idx in range(0, len(future.index)):
    arrays.append(list(future.index[idx]))
    if arrays[-1][6] == "Emissions|CO2|MAGICC Fossil and Industrial":
        arrays[-1][6] = "Emissions|CO2|Energy and Industrial Processes"
    else:
        arrays[-1][6] = "Emissions|CO2|AFOLU"
    arrays[-1][5] = "GtCO2/yr"

new_index = pd.MultiIndex.from_tuples(
    list(zip(*list(map(list, zip(*arrays))))), names=future.index.names
)
future.index = new_index
# future = future.convert_unit('GtCO2/yr')

overrides = pd.DataFrame(
    [
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        {
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },
    ]
)

harmonisation_year = 2021

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scenarios_harmonised = [
        aneris.convenience.harmonise_all(
            msdf,
            history=history,
            harmonisation_year=harmonisation_year,
            overrides=overrides,
        )
        for _, msdf in tqdm(future.groupby(["model", "scenario"]))
    ]

scenarios_harmonised = pd.concat(scenarios_harmonised).reset_index()

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

scenarios_harmonised.to_csv(
    "../../../../../data/emissions/co2_ffi_afolu_harmonized.csv", index=False
)

history = history.reset_index()

# fill emissions
for scenario in scenarios:
    data_his = history.loc[
        (history["scenario"] == "Global Carbon Project")
        & (history["variable"] == "Emissions|CO2|Energy and Industrial Processes"),
        1750:2020,
    ].values.squeeze()
    data_fut = scenarios_harmonised.loc[
        (scenarios_harmonised["scenario"] == scenario)
        & (
            scenarios_harmonised["variable"]
            == "Emissions|CO2|Energy and Industrial Processes"
        ),
        2021:2499,
    ].values.squeeze()
    data = np.concatenate((data_his, data_fut))
    fill(
        f.emissions,
        data,
        config="unspecified",
        scenario=scenario,
        #        timepoints=np.arange(1750.5, 2021),
        specie="CO2 FFI",
    )

    data_his = history.loc[
        (history["scenario"] == "Global Carbon Project")
        & (history["variable"] == "Emissions|CO2|AFOLU"),
        1750:2020,
    ].values.squeeze()
    data_fut = scenarios_harmonised.loc[
        (scenarios_harmonised["scenario"] == scenario)
        & (scenarios_harmonised["variable"] == "Emissions|CO2|AFOLU"),
        2021:2499,
    ].values.squeeze()
    data = np.concatenate((data_his, data_fut))
    fill(
        f.emissions,
        data,
        config="unspecified",
        scenario=scenario,
        #        timepoints=range(2021.5, 2501),
        specie="CO2 AFOLU",
    )
    # f.emissions.loc[dict(scenario=scenario, timepoints=)]

f.emissions.to_netcdf(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssp_emissions_1750-2500.nc"
)
