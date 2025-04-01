#!/usr/bin/env python
# coding: utf-8

"""Harmonize SSPs - we only want this for demo"""

# These harmonized files are for checking that the reconstructed historic to future
# emissions are roughly right when projected forward.

# 2024 is the last year where all emissions are available so we harmonize to this.
# this is a bit complex; we have to harmonize to the harmomized file that contains
# the running means; NOT the actual history, then when the harmonization is done
# we sub the historical back in.

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

harmonization_year = 2024

print("Making harmonized scenario files...")

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
    f"../../../../../data/emissions/"
    "historical_emissions_1750-2024.csv"
)
variables = list(df_in["variable"])
units = list(df_in["unit"])
var_units = {var: unit for var, unit in zip(variables, units)}

times = []
years = range(1750, harmonization_year + 1)
for year in years:
    times.append(datetime.datetime(year, 1, 1))
    # they are really midyears, but we just want this to work

times_future = []
years_future = range(harmonization_year, 2501)
for year in years_future:
    times_future.append(datetime.datetime(year, 1, 1))

times_harmonization = []
years_harmonization = range(2014, harmonization_year + 1)
for year in years_harmonization:
    times_harmonization.append(datetime.datetime(year, 1, 1))


history = (
    scmdata.ScmRun(
        f"../../../../../data/emissions/"
        "historical_emissions_1750-2024.csv",
        lowercase_cols=True,
    )
    .filter(region="World", variable=variables)
    .interpolate(target_times=times)
    .timeseries(time_axis="year")
)

# temporarily substitute in the 2024 harmonization value
harmonization = (
    scmdata.ScmRun(
        f"../../../../../data/emissions/"
        "historical_harmonization_5yr_running_means_2014-2024.csv",
        lowercase_cols=True,
    )
    .filter(region="World", variable=variables)
    .interpolate(target_times=times_harmonization)
    .timeseries(time_axis="year")
)

#harmonization_df = pd.read_csv(
#    f"../../../../../data/emissions/"
#    "historical_harmonization_5yr_running_means_2014-2024.csv"
#)

history_original = history.copy()
history[harmonization_year] = harmonization[harmonization_year]

print(history)
print(harmonization[harmonization_year])


future = (
    scmdata.ScmRun(
        "../../../../../data/emissions/rcmip-5-1-0-corrected-nox.csv",
        lowercase_cols=True,
    )
    .filter(scenario=scenarios, region="World")
    .interpolate(times_future)
    .timeseries(time_axis="year")
)

# de-daft units
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

# drop 1202
future = future.drop('Emissions|Halon-1202', level='variable')

# Rename variables in future emissions file
variable_mapping = {f'Emissions|{variable}': variable for variable in variables}
variable_mapping['Emissions|CO2|AFOLU'] = 'CO2 AFOLU'
variable_mapping['Emissions|CO2|Energy and Industrial Processes'] = 'CO2 FFI'

# harmonize unit
history = history.rename(index={'kt HFC43-10/yr': 'kt HFC4310mee/yr'})
history_original = history_original.rename(index={'kt HFC43-10/yr': 'kt HFC4310mee/yr'})
future = future.rename(index=variable_mapping)
future = future.rename(index={'Mt CO2/yr': 'Gt CO2/yr', 'kt N2O/yr': 'Mt N2O/yr'})


print(future.head(54))

# start the harmonization
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
            "variable": "C2F6",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "C6F14",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "CF4",
        },  # high historical variance (cov=11.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "CO",
        },  # high historical variance (cov=15.4)
#        {
#            "method": "reduce_ratio_2080",
#            "variable": "Emissions|CO2",
#        },  # always ratio method by choice
        {
            "method": "reduce_offset_2150_cov",
            "variable": "CO2 AFOLU",
        },  # high historical variance, but using offset method to prevent diff from
        # increasing when going negative rapidly (cov=23.2)
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "CO2 FFI",
        },
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
        # depending on the decision tree in aneris/method.py
        {
            "method": "constant_ratio",
            "variable": "HFC-125",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "HFC-134a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "HFC-143a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "HFC-227ea",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "HFC-23",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "HFC-32",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "HFC-4310mee",
        },  # minor f-gas with low model reporting confidence
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
        # depending on the decision tree in aneris/method.py
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "SF6",
        },  # minor f-gas with low model reporting confidence
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
        # depending on the decision tree in aneris/method.py
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "VOC",
        },  # high historical variance (cov=12.0)
    ]
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scenarios_harmonised = [
        aneris.convenience.harmonise_all(
            msdf,
            history=history,
            harmonisation_year=harmonization_year,
            overrides=overrides,
        ).reset_index(level=(5, 6, 7, 8, 9), drop=True)
        for _, msdf in tqdm(future.groupby(["model", "scenario"]), disable=1 - progress)
    ]
# reset_index is needed above because aneris for some reason gives us two copies of
# the MultiIndex

scenarios_harmonised = pd.concat(scenarios_harmonised).reset_index()

# now substitute the historical back in for 2024 and stitch historical and future together 
scenarios_harmonised = (
    scmdata.ScmRun(scenarios_harmonised)
    .interpolate(target_times=times_future[1:])
    .timeseries(time_axis="year")
)
#scenarios_harmonised_naked = scenarios_harmonised.droplevel(('model'), axis=0)

history_naked = history_original.droplevel(('model', 'scenario'), axis=0)
combined_harmonised = history_naked.join(scenarios_harmonised).reorder_levels(("model", "scenario", "region", "variable", "unit")).sort_values(["scenario", "variable"])

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

combined_harmonised.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssps_harmonized_1750-2499.csv",
)
