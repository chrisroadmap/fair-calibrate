#!/usr/bin/env python
# coding: utf-8

"""Check and run carbon cycle calibrations."""
#
# Carbon cycle tunings for 11 C4MIP models are from FaIR 2.0 paper (Leach et al. 2021),
# calibrated on CMIP6 1pct runs. Let's see if they give reasonable concentrations in
# emissions-driven mode.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.interface import fill, initialise
from fair.structure.units import compound_convert

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use("../../../../../defaults.mplstyle")


print("Making carbon cycle calibrations...")

compound_convert["CO2"]["C"]


assert fair_v == __version__


# NB: rU and rA are in GtC units, we need to convert to GtCO2
data = np.array(
    [
        [
            36.73854601035055,
            25.589821019851797,
            40.704695982343765,
            38.09182601398885,
            35.70573492682388,
            34.26732262345922,
            32.223599635483424,
            33.39478016647172,
            33.33937342916488,
            40.735872526405046,
            37.91594456570033,
        ],
        [
            0.0349535801301073,
            0.00597614250950862,
            0.010664893971021883,
            0.0005810081769186404,
            -0.005958784801017192,
            0.021861410870304354,
            0.016608701817126814,
            0.013104461258272693,
            0.031043773610946346,
            0.009471296196005063,
            0.020138272127751655,
        ],
        [
            3.036651884848311,
            5.196160258410032,
            1.2786398011433562,
            2.472206604249436,
            -0.10385375927186047,
            4.855081881723322,
            1.0693983052255476,
            3.4644393974775767,
            1.499323874009292,
            1.5631932779473914,
            2.6714005898495543,
        ],
        [
            -0.0006603263192310749,
            0.004393751681079472,
            0.004211308668836011,
            0.009783189726962682,
            0.018116906645659014,
            -0.004242277713558451,
            0.012336113500092338,
            0.003993779169272571,
            -0.002570300844565665,
            0.004887468785878646,
            0.0018119017134572424,
        ],
    ]
)
data[1, :] = data[1, :] * compound_convert["CO2"]["C"]
data[3, :] = data[3, :] * compound_convert["CO2"]["C"]

models = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CESM2",
    "CNRM-ESM2-1",
    "CanESM5",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MIROC-ES2L",
    "MPI-ESM1-2-LR",
    "NorESM2-LM",
    "UKESM1-0-LL",
]


params = pd.DataFrame(data.T, columns=["r0", "rU", "rT", "rA"], index=models)


kde = scipy.stats.gaussian_kde(params.T)
cc_sample = kde.resample(size=int(samples), seed=2421911)

mask = np.all(np.isnan(cc_sample), axis=0)
cc_sample = cc_sample[:, ~mask]
cc_sample_df = pd.DataFrame(
    data=cc_sample[:, :samples].T, columns=["r0", "rU", "rT", "rA"]
)
cc_sample_df

# ## First thing we'll do is run 11 ESM simulations, CO2 only, default forcing etc.
#
# The CO2 concentrations and warming from these simulations will not be perfect because
# of the absence of other forcers. And the individual model values of CO2 ERF are not
# taken into account either.

f = FAIR()
f.define_time(1750, 2500, 1)
f.define_scenarios(["ssp119", "ssp245", "ssp585"])
f.define_configs(models)

species = ["CO2", "CH4", "N2O"]

properties = {
    "CO2": {
        "type": "co2",
        "input_mode": "emissions",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "CH4": {
        "type": "ch4",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "N2O": {
        "type": "n2o",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
}

f.define_species(species, properties)
f.allocate()
f.fill_species_configs()
f.fill_from_rcmip()

fill(f.concentration, 729.2, specie="CH4")
fill(f.concentration, 270.1, specie="N2O")

for model in models:
    fill(
        f.species_configs["iirf_0"], params.loc[model, "r0"], specie="CO2", config=model
    )
    fill(
        f.species_configs["iirf_uptake"],
        params.loc[model, "rU"],
        specie="CO2",
        config=model,
    )
    fill(
        f.species_configs["iirf_airborne"],
        params.loc[model, "rA"],
        specie="CO2",
        config=model,
    )
    fill(
        f.species_configs["iirf_temperature"],
        params.loc[model, "rT"],
        specie="CO2",
        config=model,
    )

df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "4xCO2_cummins_ebm3_cmip6.csv"
)
models_runs = {
    "ACCESS-ESM1-5": "r1i1p1f1",
    "BCC-CSM2-MR": "r1i1p1f1",
    "CESM2": "r1i1p1f1",
    "CNRM-ESM2-1": "r1i1p1f2",
    "CanESM5": "r1i1p2f1",
    "GFDL-ESM4": "r1i1p1f1",
    "IPSL-CM6A-LR": "r1i1p1f1",
    "MIROC-ES2L": "r1i1p1f2",
    "MPI-ESM1-2-LR": "r1i1p1f1",
    "NorESM2-LM": "r1i1p1f1",
    "UKESM1-0-LL": "r1i1p1f2",
}

fill(f.climate_configs["stochastic_run"], False)

for imod, model in enumerate(models_runs):
    run = models_runs[model]
    condition = (df["model"] == model) & (df["run"] == run)
    fill(
        f.climate_configs["ocean_heat_capacity"],
        df.loc[condition, "C1":"C3"].values.squeeze(),
        config=model,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        df.loc[condition, "kappa1":"kappa3"].values.squeeze(),
    )
    fill(
        f.climate_configs["deep_ocean_efficacy"], df.loc[condition, "epsilon"].values[0]
    )
    fill(
        f.climate_configs["gamma_autocorrelation"], df.loc[condition, "gamma"].values[0]
    )

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()


# if plots:
#     custom_cycler = (
#         cycler(color=['red','darkorange','yellow','yellowgreen','green','turquoise',
#         'teal','blue','blueviolet','purple','pink'])
#     )
#
#     # these don't seem to agree with the esm-hist runs I did in another repo. v2
#     # should be a recalibration.
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     ax.plot(f.timebounds, f.concentration.loc[dict(specie='CO2', scenario='ssp585')],
#     label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     ax.plot(f.timebounds, f.temperature.loc[dict(layer=0, scenario='ssp585')],
#     label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     pl.plot(f.timebounds, f.airborne_fraction.loc[dict(specie='CO2',
#     scenario='ssp585')], label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     ax.plot(f.timebounds, f.concentration.loc[dict(specie='CO2', scenario='ssp245')],
#     label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     ax.plot(f.timebounds, f.temperature.loc[dict(layer=0, scenario='ssp245')],
#     label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     pl.plot(f.timebounds, f.airborne_fraction.loc[dict(specie='CO2',
#     scenario='ssp245')], label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     ax.plot(f.timebounds, f.concentration.loc[dict(specie='CO2',
#     scenario='ssp119')], label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     ax.plot(f.timebounds, f.temperature.loc[dict(layer=0, scenario='ssp119')],
#     label=models);
#     pl.legend()
#     pl.close()
#
#     fig, ax = pl.subplots()
#     ax.set_prop_cycle(custom_cycler)
#     pl.plot(f.timebounds, f.airborne_fraction.loc[dict(specie='CO2',
#     scenario='ssp119')], label=models);
#     pl.legend()
#     pl.close()


os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

cc_sample_df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "carbon_cycle.csv",
    index=False,
)
