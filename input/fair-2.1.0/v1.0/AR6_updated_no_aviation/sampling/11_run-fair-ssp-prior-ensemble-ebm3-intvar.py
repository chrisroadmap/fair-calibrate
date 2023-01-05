#!/usr/bin/env python
# coding: utf-8

# # AR6 calibration of FaIR 2.1
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
# We have to do this slightly differently to the examples so far. 1.5 million ensemble members is going to take up too much memory, so we run in batches of 1000, initialising a new FaIR instance for each batch, and saving the output as we go.

# ## Basic imports

# In[ ]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scipy.stats
from tqdm.auto import tqdm
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
from fair.forcing.ghg import meinshausen2020

from dotenv import load_dotenv
from fair import __version__

import warnings # remove in v2.1.1

# Get environment variables
load_dotenv()

cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
samples = 4000 #int(os.getenv("PRIOR_SAMPLES"))
batch_size = int(os.getenv("BATCH_SIZE"))


assert fair_v == __version__


# ## Set up problem

# In[ ]:


erf_2co2 = meinshausen2020(
    np.array([554.30, 731.41, 273.87]) * np.ones((1, 1, 1, 3)),
    np.array([277.15, 731.41, 273.87]) * np.ones((1, 1, 1, 3)),
    np.array((1.05, 0.86, 1.07)) * np.ones((1, 1, 1, 1)),
    np.ones((1, 1, 1, 3)),
    np.array([True, False, False]),
    np.array([False, True, False]),
    np.array([False, False, True]),
    np.array([False, False, False])
).squeeze()[0]
erf_2co2


# In[ ]:


scenarios = ['ssp245']


# In[ ]:


df_forc = pd.read_csv('../../../../../data/forcing/table_A3.3_historical_ERF_1750-2019_best_estimate.csv')
df_solar = pd.read_csv('../../../../../data/forcing/solar_erf_timebounds.csv', index_col='year')
df_volcanic = pd.read_csv('../../../../../data/forcing/volcanic_ERF_monthly_-950001-201912.csv')


# In[ ]:


volcanic_forcing = np.zeros(352)
for i, year in enumerate(np.arange(1750, 2021)):
    volcanic_forcing[i] = np.mean(df_volcanic.loc[((year-1)<=df_volcanic['year'])&(df_volcanic['year']<year)].erf)
volcanic_forcing[271:281] = np.linspace(1, 0, 10) * volcanic_forcing[270]


# In[ ]:


solar_forcing = df_solar['erf'].loc[1750:2101].values


# In[ ]:


volcanic_forcing[245]


# In[ ]:


# pl.plot(volcanic_forcing)


# In[ ]:


da_emissions = xr.load_dataarray(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/ssp_emissions_1750-2500.nc')


# In[ ]:


da_emissions


# In[ ]:


species, properties = read_properties()
species.remove("Contrails")
species.remove("NOx aviation")

# ## Load in pre-calculated prior parameter sets
#
# These should also be binaries really

# In[ ]:


df_cc=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/carbon_cycle.csv')
df_cr=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/climate_response_ebm3.csv')
df_aci=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_cloud.csv')
df_ari=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_radiation.csv')
df_ozone=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv')
df_scaling=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/forcing_scaling.csv')
df_1750co2=pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/co2_concentration_1750.csv')


# In[ ]:


df_ari


# In[ ]:


df_aci


# In[ ]:


df_cc


# In[ ]:


df_cr


# In[ ]:


df_scaling


# In[ ]:


df_1750co2


# ## Generate 1.5 million ensemble members in batches of 1000

# In[ ]:


seedgen = 1355763
seedstep = 399

trend_shape = np.ones(352)
trend_shape[:271] = np.linspace(0, 1, 271)

# for all except temperature, the full time series is not important so we can save a bit of space
temp_out = np.ones((252, samples)) * np.nan
ohc_out = np.ones((samples)) * np.nan
fari_out = np.ones((samples)) * np.nan
faci_out = np.ones((samples)) * np.nan
co2_out = np.ones((samples)) * np.nan
fo3_out = np.ones((samples)) * np.nan
ecs = np.ones(samples) * np.nan
tcr = np.ones(samples) * np.nan

calibrated_f4co2_mean = df_cr['F_4xCO2'].mean()

for ibatch, batch_start in tqdm(enumerate(range(0, samples, batch_size)), total=samples/batch_size):
    batch_end = batch_start + batch_size

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, 2101, 1)
    f.define_scenarios(scenarios)
    f.define_configs(list(range(batch_start, batch_end)))
    f.define_species(species, properties)
    f.allocate()

    # emissions and forcing
    #f.fill_from_rcmip()
    da = da_emissions.loc[dict(config='unspecified', scenario='ssp245')][:351, ...]
    fe = da.expand_dims(dim=['scenario', 'config'], axis=(1,2))
    f.emissions = fe.drop('config') * np.ones((1,1,batch_size,1))

    fill(f.forcing, volcanic_forcing[:, None, None] * df_scaling.loc[batch_start:batch_end-1, 'Volcanic'].values.squeeze(), specie='Volcanic')
    fill(f.forcing,
         solar_forcing[:, None, None] *
         df_scaling.loc[batch_start:batch_end-1, 'solar_amplitude'].values.squeeze() +
         trend_shape[:, None, None] * df_scaling.loc[batch_start:batch_end-1, 'solar_trend'].values.squeeze(),
         specie='Solar'
    )

    # climate response
    fill(f.climate_configs['ocean_heat_capacity'], df_cr.loc[batch_start:batch_end-1, 'c1':'c3'].values)
    fill(f.climate_configs['ocean_heat_transfer'], df_cr.loc[batch_start:batch_end-1, 'kappa1':'kappa3'].values)
    fill(f.climate_configs['deep_ocean_efficacy'], df_cr.loc[batch_start:batch_end-1, 'epsilon'].values.squeeze())
    fill(f.climate_configs['gamma_autocorrelation'], df_cr.loc[batch_start:batch_end-1, 'gamma'].values.squeeze())
    fill(f.climate_configs['sigma_eta'], df_cr.loc[batch_start:batch_end-1, 'sigma_eta'].values.squeeze())
    fill(f.climate_configs['sigma_xi'], df_cr.loc[batch_start:batch_end-1, 'sigma_xi'].values.squeeze())
    fill(f.climate_configs['seed'], np.arange(seedgen+batch_start*seedstep, seedgen+batch_end*seedstep, seedstep, dtype=int))
    fill(f.climate_configs['stochastic_run'], True)
    fill(f.climate_configs['use_seed'], True)
    fill(f.climate_configs['forcing_4co2'], df_cr.loc[batch_start:batch_end-1,'F_4xCO2'])

    # species level
    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs['iirf_0'], df_cc.loc[batch_start:batch_end-1, 'r0'].values.squeeze(), specie='CO2')
    fill(f.species_configs['iirf_airborne'], df_cc.loc[batch_start:batch_end-1, 'rA'].values.squeeze(), specie='CO2')
    fill(f.species_configs['iirf_uptake'], df_cc.loc[batch_start:batch_end-1, 'rU'].values.squeeze(), specie='CO2')
    fill(f.species_configs['iirf_temperature'], df_cc.loc[batch_start:batch_end-1, 'rT'].values.squeeze(), specie='CO2')

    # aerosol indirect
    fill(f.species_configs['aci_scale'], df_aci.loc[batch_start:batch_end-1, 'beta'].values.squeeze())
    fill(f.species_configs['aci_shape'], df_aci.loc[batch_start:batch_end-1, 'shape_so2'].values.squeeze(), specie='Sulfur')
    fill(f.species_configs['aci_shape'], df_aci.loc[batch_start:batch_end-1, 'shape_bc'].values.squeeze(), specie='BC')
    fill(f.species_configs['aci_shape'], df_aci.loc[batch_start:batch_end-1, 'shape_oc'].values.squeeze(), specie='OC')

    # methane lifetime baseline
    fill(f.species_configs['unperturbed_lifetime'], 10.11702748, specie='CH4')

    # emissions adjustments for N2O and CH4 (we don't want to make these defaults as people might wanna run pulse expts with these gases)
    fill(f.species_configs['baseline_emissions'], 19.019783117809567, specie='CH4')
    fill(f.species_configs['baseline_emissions'], 0.08602230754, specie='N2O')

    # aerosol direct
    for specie in df_ari:
        fill(f.species_configs['erfari_radiative_efficiency'], df_ari.loc[batch_start:batch_end-1, specie], specie=specie)

    # forcing
    for specie in df_scaling:
        if specie in ['minorGHG', 'solar_amplitude', 'solar_trend', 'Volcanic']:
            continue
        fill(f.species_configs['forcing_scale'], df_scaling.loc[batch_start:batch_end-1, specie].values.squeeze(), specie=specie)
    for specie in ['CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 'HCFC-22', 'HCFC-141b', 'HCFC-142b',
        'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br', 'Halon-1211', 'Halon-1202', 'Halon-1301', 'Halon-2402',
        'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14', 'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2',
        'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32',
        'HFC-365mfc', 'HFC-4310mee']:
        fill(f.species_configs['forcing_scale'], df_scaling.loc[batch_start:batch_end-1, 'minorGHG'].values.squeeze(), specie=specie)

    # ozone
    for specie in df_ozone:
        fill(f.species_configs['ozone_radiative_efficiency'], df_ozone.loc[batch_start:batch_end-1, specie], specie=specie)

    # tune down volcanic efficacy
    fill(f.species_configs['forcing_efficacy'], 0.6, specie='Volcanic')


    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(f.species_configs['baseline_concentration'], df_1750co2.loc[batch_start:batch_end-1, 'co2_concentration'].values.squeeze(), specie='CO2')

    # initial conditions
    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    # at this point dump out some batch output and put the constraining in another notebook
    temp_out[:, batch_start:batch_end] = f.temperature[100:, 0, :, 0]
    ohc_out[batch_start:batch_end] = f.ocean_heat_content_change[268:270, 0, :].mean(axis=0)-f.ocean_heat_content_change[221:223, 0, :].mean(axis=0)
    co2_out[batch_start:batch_end] = f.concentration[264, 0, :, 2]
    fari_out[batch_start:batch_end] = np.average(f.forcing[255:266, 0, :, 56], weights=np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]), axis=0)
    faci_out[batch_start:batch_end] = np.average(f.forcing[255:266, 0, :, 57], weights=np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]), axis=0)
    fo3_out[batch_start:batch_end] = f.forcing[269:271, 0, :, 58].mean(axis=0)
    ecs[batch_start:batch_end] = f.ebms.ecs
    tcr[batch_start:batch_end] = f.ebms.tcr


# In[ ]:


f.ebms.ecs


# In[ ]:


#2 * erf_2co2 * (1 + 0.561*(calibrated_f4co2_mean - df_cr.loc[:100,'F_4xCO2'])/calibrated_f4co2_mean)


# In[ ]:


# # use F2xCO2 and scale factor to determine ECS: this is what each climate simulation actually "sees"
# calibrated_f4co2_mean = df_cr['F_4xCO2'].mean()

# ecs = np.ones(samples) * np.nan
# tcr = np.ones(samples) * np.nan
# for iconf in tqdm(range(samples)):
#     ebm = EnergyBalanceModel(
#         ocean_heat_capacity = np.array([df_cr.loc[iconf,'c1'], df_cr.loc[iconf,'c2'], df_cr.loc[iconf, 'c3']]),
#         ocean_heat_transfer = np.array([df_cr.loc[iconf,'kappa1'], df_cr.loc[iconf,'kappa2'], df_cr.loc[iconf,'kappa3']]),
#         deep_ocean_efficacy = df_cr.loc[iconf,'epsilon'],
#         forcing_4co2 = 2 * erf_2co2 * (1+ 0.561*(calibrated_f4co2_mean - df_cr.loc[iconf,'F_4xCO2'])/calibrated_f4co2_mean) #2*erf_2co2*df_scale.loc[i,'CO2'],
#     )
#     ebm.emergent_parameters()
#     ecs[iconf] = ebm.ecs
#     tcr[iconf] = ebm.tcr


# In[ ]:


# pl.plot(f.temperature[100:, 0, :, 0]);


# In[ ]:


# pl.plot(f.ocean_heat_content_change[100:, 0, :]);


# In[ ]:


# pl.plot(f.concentration[100:, 0, :, 2]);


# In[ ]:


# pl.plot(f.concentration[100:, 0, :, 3]);


# In[ ]:


# pl.plot(f.forcing[:, 0, :, 2]);


# In[ ]:


# pl.plot(f.forcing[:, 0, :, 55]);


# In[ ]:


# pl.plot(f.forcing[:, 0, :, 56]);


# In[ ]:


# pl.plot(f.forcing[:, 0, :, 57]);


# In[ ]:


# pl.plot(f.forcing.loc[dict(specie='Contrails', scenario='ssp245')]);


# In[ ]:


# pl.plot(f.forcing.loc[dict(specie='Light absorbing particles on snow and ice', scenario='ssp245')]);


# In[ ]:


# pl.plot(f.forcing.loc[dict(specie='Land use', scenario='ssp245')]);


# In[ ]:


# pl.plot(f.forcing.loc[dict(specie='Stratospheric water vapour', scenario='ssp245')]);


# In[ ]:


# pl.plot(f.forcing.loc[dict(specie='Solar', scenario='ssp245')]);


# In[ ]:


# pl.plot(f.forcing.loc[dict(specie='Volcanic', scenario='ssp245')]);


# In[ ]:


# pl.plot(f.forcing_sum.loc[dict(scenario='ssp245')]);


# In[ ]:


os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/', exist_ok=True)


# In[ ]:


np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy', temp_out, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ocean_heat_content_2018_minus_1971.npy', ohc_out, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/concentration_co2_2014.npy', co2_out, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ari_2005-2014_mean.npy', fari_out, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_aci_2005-2014_mean.npy', faci_out, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ozone_2019.npy', fo3_out, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy', ecs, allow_pickle=True)
np.save(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy', tcr, allow_pickle=True)


# In[ ]:


# pl.hist(ohc_out)
# np.percentile(ohc_out, (5, 50, 95))


# In[ ]:


# pl.hist(co2_out)
# np.percentile(co2_out, (5, 50, 95))


# In[ ]:


# pl.hist(fari_out)
# np.percentile(fari_out, (5, 50, 95))


# In[ ]:


# pl.hist(faci_out)
# np.percentile(faci_out, (5, 50, 95))


# In[ ]:


# pl.hist(fo3_out)
# np.percentile(fo3_out, (5, 50, 95))


# In[ ]:


# pl.hist(ecs)
# np.percentile(ecs, (5, 50, 95))


# In[ ]:


# pl.hist(tcr)
# np.percentile(tcr, (5, 50, 95))


# In[ ]:
