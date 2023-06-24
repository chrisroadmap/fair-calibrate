#!/usr/bin/env python
# coding: utf-8

"""Make concentration binary file."""

# SSPs from RCMIP

import os

import pandas as pd
import pooch
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.io import read_properties

load_dotenv()


cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

print("Making SSP concentration binary...")

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
#species.remove('CO2 FFI')
species.remove('Halon-1202')
#del properties['CO2 FFI']
del properties['Halon-1202']
for specie in species:
    if properties[specie]['greenhouse_gas']:
        properties[specie]['input_mode'] = 'concentration'

f = FAIR()
f.define_time(1750, 2500, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()
f.fill_from_rcmip()

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/concentration/",
    exist_ok=True,
)

f.concentration.to_netcdf(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/concentration/"
    "ssp_concentration_1750-2500.nc"
)
