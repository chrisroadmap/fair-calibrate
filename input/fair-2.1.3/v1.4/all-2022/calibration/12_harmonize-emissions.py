#!/usr/bin/env python
# coding: utf-8

"""Make harmonized emissions to SSPs then emissions binary file."""

# These harmonized files are for checking that the reconstructed historic to future
# emissions are roughly right when projected forward using the SSPs.

# We multiply the PRIMAP emissions by their scale factors previously calculated.

# We don't have future projections from NOx aviation, so we drop it.

import datetime
import os
import warnings

import aneris.convenience
import numpy as np
import pandas as pd
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
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__

print("Making harmonized SSP emissions binary...")

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
species.remove("Halon-1202")
species.remove("NOx aviation")
species.remove("Contrails")

f = FAIR(ch4_method="thornhill2021")
f.define_time(1750, 2500, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()
f.fill_from_rcmip()

df_in = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_1750-2022.csv"
)
variables = list(df_in["Variable"])
units = list(df_in["Unit"])
var_units = {var: unit for var, unit in zip(variables, units)}

times = []
years = range(1750, 2023)
for year in years:
    times.append(datetime.datetime(year, 1, 1))
    # they are really midyears, but we just want this to work

times_future = []
years_future = range(2022, 2501)
for year in years_future:
    times_future.append(datetime.datetime(year, 1, 1))

# get scale factors
scale_factors = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "emissions_scalings.csv",
    index_col=0,
)

history = (
    scmdata.ScmRun(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
        "all_1750-2022.csv",
        lowercase_cols=True,
    )
    .filter(region="World", variable=variables)
    .interpolate(target_times=times)
    .timeseries(time_axis="year")
)

# Apply emissions scalings to historical
for sf in scale_factors:
    history.iloc[history.index.get_level_values("variable") == f"Emissions|{sf}"] = (
        history.iloc[history.index.get_level_values("variable") == f"Emissions|{sf}"]
        * scale_factors.loc["historical_best", sf]
    )
history.reorder_levels(
    ["model", "scenario", "region", "variable", "unit"]
).sort_index().to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_scaled_1750-2022.csv"
)

arrays = []
for idx in range(0, len(history.index)):
    arrays.append(list(history.index[idx]))
    arrays[-1][2] = "GCP+CEDS+PRIMAP+GFED"

for iu, unit in enumerate(units):
    arrays[iu][3] = unit

new_index = pd.MultiIndex.from_tuples(
    list(zip(*list(map(list, zip(*arrays))))), names=history.index.names
)
history.index = new_index

future = (
    scmdata.ScmRun(
        "../../../../../data/emissions/rcmip-5-1-0-corrected-nox.csv",
        lowercase_cols=True,
    )
    .filter(scenario=scenarios, variable=variables, region="World")
    .interpolate(times_future)
    .timeseries(time_axis="year")
)

future.iloc[
    future.index.get_level_values("variable").isin(
        (
            "Emissions|CO2|Energy and Industrial Processes",
            "Emissions|CO2|AFOLU",
            "Emissions|N2O",
        )
    )
] = (
    future.iloc[
        future.index.get_level_values("variable").isin(
            (
                "Emissions|CO2|Energy and Industrial Processes",
                "Emissions|CO2|AFOLU",
                "Emissions|N2O",
            )
        )
    ]
    / 1000
)

arrays = []
for idx in range(0, len(future.index)):
    arrays.append(list(future.index[idx]))

for iu in range(len(future.index)):
    arrays[iu][3] = var_units[arrays[iu][4]]

new_index = pd.MultiIndex.from_tuples(
    list(zip(*list(map(list, zip(*arrays))))), names=future.index.names
)
future.index = new_index

history = history.reorder_levels(
    ["model", "scenario", "region", "variable", "unit"]
).sort_index()
future = future.reorder_levels(
    ["model", "scenario", "region", "variable", "unit"]
).sort_index()

# Harmonization overrides - use same as RCMIP
overrides = pd.DataFrame(
    [
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|C2F6",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|C6F14",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CF4",
        },  # high historical variance (cov=11.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CO",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2080",
            "variable": "Emissions|CO2",
        },  # always ratio method by choice
        {
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },  # high historical variance, but using offset method to prevent diff from
        # increasing when going negative rapidly (cov=23.2)
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
        # depending on the decision tree in aneris/method.py
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-125",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-134a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-143a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-227ea",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-23",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-32",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC-4310mee",
        },  # minor f-gas with low model reporting confidence
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
        # depending on the decision tree in aneris/method.py
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "Emissions|SF6",
        },  # minor f-gas with low model reporting confidence
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
        # depending on the decision tree in aneris/method.py
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
        },  # high historical variance (cov=12.0)
    ]
)

harmonisation_year = 2022

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scenarios_harmonised = [
        aneris.convenience.harmonise_all(
            msdf,
            history=history,
            harmonisation_year=harmonisation_year,
            overrides=overrides,
        ).reset_index(level=(5, 6, 7, 8, 9), drop=True)
        for _, msdf in tqdm(future.groupby(["model", "scenario"]), disable=1 - progress)
    ]
# reset_index is needed above because aneris for some reason gives us two copies of
# the MultiIndex

scenarios_harmonised = pd.concat(scenarios_harmonised).reset_index()

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

scenarios_harmonised.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssps_harmonized_2022-2499.csv",
    index=False,
)


history = history.reset_index()

fair_map = {var: var.split("|")[-1] for var in variables}
fair_map["Emissions|CO2|Energy and Industrial Processes"] = "CO2 FFI"
fair_map["Emissions|CO2|AFOLU"] = "CO2 AFOLU"

# fill emissions
for scenario in scenarios:
    for specie in fair_map:
        data_his = history.loc[
            (history["scenario"] == "GCP+CEDS+PRIMAP+GFED")
            & (history["variable"] == specie),
            1750:2021,
        ].values.squeeze()
        data_fut = scenarios_harmonised.loc[
            (scenarios_harmonised["scenario"] == scenario)
            & (scenarios_harmonised["variable"] == specie),
            2022:2499,
        ].values.squeeze()
        data = np.concatenate((data_his, data_fut))
        fill(
            f.emissions,
            data,
            config="unspecified",
            scenario=scenario,
            #        timepoints=np.arange(1750.5, 2021),
            specie=fair_map[specie],
        )

f.emissions.to_netcdf(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssps_harmonized_1750-2499.nc"
)
