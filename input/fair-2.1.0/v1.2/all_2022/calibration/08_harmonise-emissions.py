#!/usr/bin/env python
# coding: utf-8

"""Make harmonized GCP CO2 emissions to SSPs then emissions binary file."""

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
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

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
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|Sulfur",
    "Emissions|CO",
    "Emissions|VOC",
    "Emissions|NOx",
    "Emissions|BC",
    "Emissions|OC",
    "Emissions|NH3",
]

var_units = {
    "Emissions|CO2|Energy and Industrial Processes": "Gt CO2/yr",
    "Emissions|CO2|AFOLU": "Gt CO2/yr",
    "Emissions|CH4": "Mt CH4/yr",
    "Emissions|N2O": "Mt N2O/yr",
    "Emissions|Sulfur": "Mt SO2/yr",
    "Emissions|CO": "Mt CO/yr",
    "Emissions|VOC": "Mt VOC/yr",
    "Emissions|NOx": "Mt NO2/yr",
    "Emissions|BC": "Mt BC/yr",
    "Emissions|OC": "Mt OC/yr",
    "Emissions|NH3": "Mt NH3/yr",
}

##### ALSO NEED NEW SCRIPT THAT CALIBRATES THE N2O LIFETIME FROM EMISSIONS
##### and to check that NF3 and SF6 are sensible

rcmip_variables = [
    "Emissions|CO2|MAGICC Fossil and Industrial",
    "Emissions|CO2|MAGICC AFOLU",
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|Sulfur",
    "Emissions|CO",
    "Emissions|VOC",
    "Emissions|NOx",
    "Emissions|BC",
    "Emissions|OC",
    "Emissions|NH3",
]

times = []
years = range(1750, 2022)
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
others = (
    scmdata.ScmRun(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
        "primap_ceds_gfed_1750-2022.csv", lowercase_cols=True
    )
    .filter(region="World", variable=variables)
    .interpolate(target_times=times)
    .timeseries(time_axis="year")
)
history = pd.concat((history, others))

# I don't like scmdata's default conversion, so hack
# history["units"] = "Gt CO2/yr"
history.iloc[history.index.get_level_values('variable').isin((
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO2|AFOLU"
))] = history.iloc[history.index.get_level_values('variable').isin((
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO2|AFOLU",
))] * 44.009 / 12.011
arrays = []
for idx in range(0, len(history.index)):
    arrays.append(list(history.index[idx]))
    arrays[-1][2] = "GCP+PRIMAP+CEDS+GFED"

units = [
    "Gt CO2/yr",
    "Gt CO2/yr",
    "Mt CH4/yr",
    "Mt N2O/yr",
    "Mt SO2/yr",
    "Mt CO/yr",
    "Mt VOC/yr",
    "Mt NO2/yr",
    "Mt BC/yr",
    "Mt OC/yr",
    "Mt NH3/yr"
]
for iu, unit in enumerate(units):
    arrays[iu][3] = unit

new_index = pd.MultiIndex.from_tuples(
    list(zip(*list(map(list, zip(*arrays))))), names=history.index.names
)
history.index = new_index

future = (
    scmdata.ScmRun("../../../../../data/emissions/rcmip-5-1-0-corrected-nox.csv", lowercase_cols=True)
    .filter(scenario=scenarios, variable=rcmip_variables, region="World")
    .interpolate(times_future)
    .timeseries(time_axis="year")
)

#future[:, 0] = future[:,0] / 1000
future.iloc[future.index.get_level_values('variable').isin((
    "Emissions|CO2|MAGICC Fossil and Industrial",
    "Emissions|CO2|MAGICC AFOLU",
    "Emissions|N2O"
))] = future.iloc[future.index.get_level_values('variable').isin((
    "Emissions|CO2|MAGICC Fossil and Industrial",
    "Emissions|CO2|MAGICC AFOLU",
    "Emissions|N2O"
))] / 1000


arrays = []
for idx in range(0, len(future.index)):
    arrays.append(list(future.index[idx]))
    if arrays[-1][4] == "Emissions|CO2|MAGICC Fossil and Industrial":
        arrays[-1][4] = "Emissions|CO2|Energy and Industrial Processes"
    elif arrays[-1][4] == "Emissions|CO2|MAGICC AFOLU":
        arrays[-1][4] = "Emissions|CO2|AFOLU"

for iu in range(len(future.index)):
    arrays[iu][3] = var_units[arrays[iu][4]]

new_index = pd.MultiIndex.from_tuples(
    list(zip(*list(map(list, zip(*arrays))))), names=future.index.names
)
future.index = new_index
# future = future.convert_unit('GtCO2/yr')

overrides = pd.DataFrame(
    [
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CO",
        },
        {
            "method": "reduce_ratio_2080",
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        {
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
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
        for _, msdf in tqdm(future.groupby(["model", "scenario"]), disable=1-progress)
    ]

scenarios_harmonised = pd.concat(scenarios_harmonised).reset_index()

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

scenarios_harmonised.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/co2_ch4_n2o_slcfs_harmonized.csv", index=False
)


history = history.reset_index()

fair_map = {
    "Emissions|CO2|Energy and Industrial Processes": "CO2 FFI",
    "Emissions|CO2|AFOLU": "CO2 AFOLU",
    "Emissions|CH4": "CH4",
    "Emissions|N2O": "N2O",
    "Emissions|Sulfur": "Sulfur",
    "Emissions|CO": "CO",
    "Emissions|VOC": "VOC",
    "Emissions|NOx": "NOx",
    "Emissions|BC": "BC",
    "Emissions|OC": "OC",
    "Emissions|NH3": "NH3"
}

# fill emissions
for scenario in scenarios:
    for specie in fair_map:
        data_his = history.loc[
            (history["scenario"] == "GCP+PRIMAP+CEDS+GFED")
            & (history["variable"] == specie),
            1750:2020,
        ].values.squeeze()
        data_fut = scenarios_harmonised.loc[
            (scenarios_harmonised["scenario"] == scenario)
            & (
                scenarios_harmonised["variable"]
                == specie
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
            specie=fair_map[specie],
        )

f.emissions.to_netcdf(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssp_emissions_1750-2500.nc"
)
