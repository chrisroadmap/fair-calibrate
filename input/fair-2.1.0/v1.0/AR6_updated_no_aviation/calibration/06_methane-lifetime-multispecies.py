#!/usr/bin/env python
# coding: utf-8

# # What affects methane chemical lifetime?
#
# - methane
# - VOCs
# - NOx
# - Ozone
# - halocarbons (specifically ODSs)
# - N2O
# - climate
#
# Ozone itself is a function of other precursors: we do not include ozone as a direct influence on methane lifetime, and restrict ourselves to directly emitted anthropogenic species.
#
# Gill Thornhill published two papers on methane lifetime: one on the chemical adjustments to lifetime, and one on the climate adjustments. Both effects will be included. We will
#
# 1. take AerChemMIP multi-model means from Gill's papers
# 2. run the lifetime relationship to individual AerChemMIP models in Gill's papers
# 3. find a least squares fit with reasonable sensitivies across the historical
# 4. run a Monte Carlo that perturbs the sensitivity of lifetime to each emitted species

# In[ ]:


import os
import numpy as np
import pandas as pd
import pooch
import matplotlib.pyplot as pl
import time
import scipy.stats
import scipy.optimize
from tqdm import tqdm

from fair import FAIR
from fair.interface import fill, initialise

from dotenv import load_dotenv
from fair import __version__

# Get environment variables
load_dotenv()

cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
assert fair_v == __version__


# In[ ]:


# #pl.rcParams['figure.figsize'] = (11.4, 11.4)
# pl.rcParams['font.size'] = 16
# pl.rcParams['font.family'] = 'Arial'
# pl.rcParams['ytick.direction'] = 'in'
# pl.rcParams['ytick.minor.visible'] = True
# pl.rcParams['ytick.major.right'] = True
# pl.rcParams['ytick.right'] = True
# pl.rcParams['xtick.direction'] = 'in'
# pl.rcParams['xtick.minor.visible'] = True
# pl.rcParams['xtick.major.top'] = True
# pl.rcParams['xtick.top'] = True
# pl.rcParams['axes.spines.top'] = True
# pl.rcParams['axes.spines.bottom'] = True
# pl.rcParams['figure.dpi'] = 300


# In[ ]:


# mkdir_p('../plots/')


# ## Temperature data
#
# Use observations 1850-2020, then simulate an SSP3-7.0 climate with a linear warming rate to 4C in 2100.

# In[ ]:


df_temp = pd.read_csv('../../../../../data/forcing/AR6_GMST.csv')
gmst = np.zeros(351)
gmst[100:271] = df_temp['gmst'].values
gmst[271:351] = np.linspace(gmst[270], 4, 80)


# ## Get emissions and concentrations

# In[ ]:


rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

rcmip_concentration_file = pooch.retrieve(
    url=(
        "doi:10.5281/zenodo.4589756/"
        "rcmip-concentrations-annual-means-v5-1-0.csv"
    ),
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
)

df_emis = pd.read_csv(rcmip_emissions_file)
df_conc = pd.read_csv(rcmip_concentration_file)
input = {}
hc_input = {}


# In[ ]:


conc_species = ['CH4', 'N2O']
hc_species = ['CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 'HCFC-22', 'HCFC-141b', 'HCFC-142b',
               'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br', 'Halon-1211', 'Halon-1301', 'Halon-2402']

for species in conc_species:
    input[species] = df_conc.loc[
        (df_conc['Scenario']=='ssp370') & (df_conc['Variable'].str.endswith(species)) &
        (df_conc['Region']=='World'), '1750':'2100'
    ].interpolate(axis=1).values.squeeze()

for species in hc_species:
    species_rcmip_name = species.replace("-", "")
    hc_input[species] = df_conc.loc[
        (df_conc['Scenario']=='ssp370') & (df_conc['Variable'].str.endswith(species_rcmip_name)) &
        (df_conc['Region']=='World'), '1750':'2100'
    ].interpolate(axis=1).values.squeeze()


# In[ ]:


emis_species = ['CO', 'VOC', 'NOx']
for species in emis_species:
    input[species] = df_emis.loc[
        (df_emis['Scenario']=='ssp370') & (df_emis['Variable'].str.endswith(species)) &
        (df_emis['Region']=='World'), '1750':'2100'
    ].interpolate(axis=1).values.squeeze()


# In[ ]:


input['temp'] = gmst


# In[ ]:


def calculate_eesc(
    concentration,
    fractional_release,
    fractional_release_cfc11,
    cl_atoms,
    br_atoms,
    br_cl_ratio = 45,
):

    # EESC is in terms of CFC11-eq
    eesc_out = (
        cl_atoms * (concentration) * fractional_release / fractional_release_cfc11 +
        br_cl_ratio * br_atoms * (concentration) * fractional_release / fractional_release_cfc11
    ) * fractional_release_cfc11
    return eesc_out


# In[ ]:


fractional_release = {
    'CFC-11':0.47,
    'CFC-12':0.23,
    'CFC-113':0.29,
    'CFC-114':0.12,
    'CFC-115':0.04,
    'HCFC-22':0.13,
    'HCFC-141b':0.34,
    'HCFC-142b':0.17,
    'CCl4':0.56,
    'CHCl3':0,
    'CH2Cl2':0,
    'CH3Cl':0.44,
    'CH3CCl3':0.67,
    'CH3Br':0.6,
    'Halon-1211':0.62,
    'Halon-1301':0.28,
    'Halon-2402':0.65
}

cl_atoms = {
    'CFC-11':3,
    'CFC-12':2,
    'CFC-113':3,
    'CFC-114':2,
    'CFC-115':1,
    'HCFC-22':1,
    'HCFC-141b':2,
    'HCFC-142b':1,
    'CCl4':4,
    'CHCl3':3,
    'CH2Cl2':2,
    'CH3Cl':1,
    'CH3CCl3':3,
    'CH3Br':0,
    'Halon-1211':1,
    'Halon-1301':0,
    'Halon-2402':0
}

br_atoms = {
    'CFC-11':0,
    'CFC-12':0,
    'CFC-113':0,
    'CFC-114':0,
    'CFC-115':0,
    'HCFC-22':0,
    'HCFC-141b':0,
    'HCFC-142b':0,
    'CCl4':0,
    'CHCl3':0,
    'CH2Cl2':0,
    'CH3Cl':0,
    'CH3CCl3':0,
    'CH3Br':1,
    'Halon-1211':1,
    'Halon-1301':1,
    'Halon-2402':2
}


# In[ ]:


hc_eesc = {}
total_eesc = 0

for species in hc_species:
    hc_eesc[species] = calculate_eesc(
        hc_input[species],
        fractional_release[species],
        fractional_release['CFC-11'],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc = total_eesc + hc_eesc[species]


# In[ ]:


total_eesc_1850 = total_eesc[100]


# In[ ]:


total_eesc_1850


# In[ ]:


# hfc_erf = {}
# hfc_sum = 0
# for species in ['HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32',
#     'HFC-365mfc', 'HFC-4310mee']:
#     hfc_erf[species] = (input[species][269] * radiative_efficiency[species]/1000)
#     hfc_sum = hfc_sum + hfc_erf[species]


# In[ ]:


# hfc134a_eq = 0
# for species in hfc_species:
#     hfc134a_eq = hfc134a_eq + (hfc_input[species] * radiative_efficiency[species])/(radiative_efficiency['HFC-134a'])


# In[ ]:


#total_eesc, hc_eesc['CFC-11'], hc_eesc['CFC-12']


# In[ ]:


for species in hc_species:
    pl.plot(hc_eesc[species])


# In[ ]:


input['HC'] = total_eesc


# Use 1850 and 2014 emissions or concentrations corresponding to methane lifetime changes in Thornhill et al. 2021.
#
# Could we also take into account the fact that there are multiple loss pathways for CH4:
# - tropospheric OH loss is 560 Tg/yr
# - chlorine oxidation, 11 Tg/yr, assumed not included in AerChemMIP models
# - stratospheric loss is 31 Tg/yr, assumed not included in AerChemMIP models
# - soil uptake, 30 Tg/yr, not included in AerChemMIP models
#
# Saunois (2020): 90% of sink is OH chemistry in troposphere and is 553 [476–677] Tg CH4 yr−1, which is close to the IPCC number of 560, (chapter 5)
#
# Chapter 6 only give time constants for soil uptake and the combined chemistry loss (trop OH + chlorine + stratosphere).

# In[ ]:


def alpha_scaling_exp(
    input,
    baseline,
    normalisation,
    beta,
):
    log_lifetime_scaling = 0
    for species in ['CH4', 'N2O', 'VOC', 'HC', 'NOx', 'temp']:
        log_lifetime_scaling = log_lifetime_scaling + (
            np.log(1 + (input[species]-baseline[species])/normalisation[species] * beta[species])
        )
    return np.exp(log_lifetime_scaling)


# In[ ]:


normalisation = {}
for species in ['CH4', 'N2O', 'VOC', 'NOx', 'HC']:
    normalisation[species] = input[species][264] - input[species][100]
    print(species, normalisation[species])
normalisation['temp'] = 1


# In[ ]:


baseline = {}
for species in ['CH4', 'N2O', 'VOC', 'NOx', 'HC']:
    baseline[species] = input[species][100]
baseline['temp'] = 0


# ## Steps 1 and 2
#
# Get and tune to AerChemMIP models
#
# MRI and GISS both give pretty good historical emulations

# In[ ]:


parameters = {}

parameters['AerChemMIP_mean'] = {
    'base': 10.0,
    'CH4': +0.22,
    'NOx': -0.33,
#    'CO': 0,
    'VOC': +0.19,
    'HC': -0.037,
    'N2O': -0.02,
    'temp': -0.006,
}

parameters['UKESM'] = {
    'base': 8,
    'CH4': +0.22,
    'NOx': -0.25,
#    'CO': 0,
    'VOC': +0.11,
    'HC': -0.049,
    'N2O': -0.012,
    'temp': -0.0408
}

# we'll exclude BCC and CESM as they don't have VOC expt and that's important.
# We can live with a missing N2O from GFDL and a missing temperature feedback from MRI.

parameters['GFDL'] = {
    'base': 9.6,
    'CH4': +0.21,
    'NOx': -0.33,
#    'CO': 0,
    'VOC': +0.15,
    'HC': -0.075,
    'N2O': 0,  # missing
    'temp': -0.0205
}

parameters['GISS'] = {
    'base': 13.4,
    'CH4': +0.18,
    'NOx': -0.46,
#    'CO': 0,
    'VOC': +0.27,
    'HC': -0.006,
    'N2O': -0.039,
    'temp': -0.0333
}

parameters['MRI'] = {
    'base': 10.1,
    'CH4': +0.22,
    'NOx': -0.26,
#    'CO': 0,
    'VOC': +0.21,
    'HC': -0.024,
    'N2O': -0.013,
    'temp': 0  # missing
}


# In[ ]:


lifetime_scaling = {}


# In[ ]:


models = ['UKESM', 'GFDL', 'GISS', 'MRI']


# In[ ]:


for model in models:
    print(parameters[model])
    lifetime_scaling[model] = alpha_scaling_exp(
        input,
        baseline,
        normalisation,
        parameters[model],
    )


# In[ ]:


#pl.plot(np.arange(1750, 2501), aerchemmip_mean[:] * 8.25)


# In[ ]:


#1/(1/135 + 1/9.7)


# In[ ]:


#1/(1/120 + 1/200 + 1/150 + 1/11.2)


# In[ ]:


1/np.inf


# In[ ]:


# put this into a simple one box model
def one_box(
    emissions,
    gas_boxes_old,
    airborne_emissions_old,
    burden_per_emission,
    lifetime,
    alpha_lifetime,
    partition_fraction,
    pre_industrial_concentration,
    soil_lifetime=135,
    timestep=1,
    natural_emissions_adjustment=0,
):

    effective_lifetime = 1/(1/(alpha_lifetime * lifetime) + 1/soil_lifetime)
    decay_rate = timestep/(effective_lifetime)
    decay_factor = np.exp(-decay_rate)
    gas_boxes_new = (
        partition_fraction *
        (emissions-natural_emissions_adjustment) *
        1 / decay_rate *
        (1 - decay_factor) * timestep + gas_boxes_old * decay_factor
    )
    airborne_emissions_new = gas_boxes_new
    concentration_out = (
        pre_industrial_concentration +
        burden_per_emission * (
            airborne_emissions_new + airborne_emissions_old
        ) / 2
    )
    return concentration_out, gas_boxes_new, airborne_emissions_new


# In[ ]:


emis_ch4 = df_emis.loc[
    (df_emis['Scenario']=='ssp370') & (df_emis['Variable'].str.endswith('CH4')) &
    (df_emis['Region']=='World'), '1750':'2500'
].interpolate(axis=1).values.squeeze()


# In[ ]:


burden_per_emission = 1 / (5.1352e18 / 1e18 * 16.043 / 28.97)
partition_fraction = 1
pre_industrial_concentration = 729.2
natural_emissions_adjustment = emis_ch4[0]


# In[ ]:


conc_ch4 = {}


# In[ ]:


for model in models:
    conc_ch4[model] = np.zeros(351)
    gas_boxes = 0
    airborne_emissions = 0
    for i in range(351):
        conc_ch4[model][i], gas_boxes, airborne_emissions = one_box(
            emis_ch4[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            parameters[model]['base'],
            lifetime_scaling[model][i],
            partition_fraction,
            pre_industrial_concentration,
            soil_lifetime=np.inf,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )


# In[ ]:


# this is for another day :)
# df_ch4 = pd.read_csv('../data/calibration/methane_ukesm1.csv', index_col=0)


# In[ ]:


# df_ch4.mean(axis=1)


# In[ ]:


# pl.plot(np.arange(1850.5, 2015), conc_ch4['UKESM'][100:265], label='FaIR UKESM1 calibration')
# pl.plot(df_ch4.index, df_ch4.mean(axis=1), label='UKESM1')
# pl.legend()
# #pl.savefig('../plots/ukesm_historical_methane.png')


# In[ ]:


for model in models:
    pl.plot(np.arange(1750, 2021), conc_ch4[model][:271], label=model)
pl.plot(np.arange(1750, 2021), input['CH4'][:271], color='k', label='obs')
pl.legend()
#pl.savefig('../plots/aerchemmip_tuning_ch4_conc_1750-2020.pdf')


# ## Step 3
#
# Find least squares sensible historical fit

# In[ ]:


invect = np.array([input['CH4'], input['NOx'], input['VOC'], input['HC'], input['N2O'], input['temp']])


# In[ ]:


def fit_precursors(x, rch4, rnox, rvoc, rhc, rn2o, rtemp, rbase):
    conc_ch4 = np.zeros(271)
    gas_boxes = 0
    airborne_emissions = 0

    params = {}
    params['CH4'] = rch4
    params['NOx'] = rnox
    params['VOC'] = rvoc
    params['HC'] = rhc
    params['N2O'] = rn2o
    params['temp'] = rtemp

    inp = {}
    inp['CH4'] = x[0]
    inp['NOx'] = x[1]
    inp['VOC'] = x[2]
    inp['HC'] = x[3]
    inp['N2O'] = x[4]
    inp['temp'] = x[5]

    lifetime_scaling = alpha_scaling_exp(
        inp,
        baseline,
        normalisation,
        params,
    )

    for i in range(271):
        conc_ch4[i], gas_boxes, airborne_emissions = one_box(
            emis_ch4[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            rbase,
            lifetime_scaling[i],
            partition_fraction,
            pre_industrial_concentration,
            soil_lifetime=np.inf,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )
    return conc_ch4


p, cov = scipy.optimize.curve_fit(
    fit_precursors,
    invect[:, :271],
    input['CH4'][:271],
    bounds = (  # AerChemMIP min to max range
        (0.18, -0.46, 0.11, -0.075, -0.039, -0.0408, 6.3),
        (0.26, -0.25, 0.27, -0.006, -0.012, +0.0718, 13.4)
    )
)


# In[ ]:


parameters['best_fit'] = {
    'base': p[6],
    'CH4': p[0],
    'NOx': p[1],
#    'CO': 0,
    'VOC': p[2],
    'HC': p[3],
    'N2O': p[4],
    'temp': p[5],
}
p


# In[ ]:


# these are the feedback values per ppb / per Mt that go into FaIR
for specie in ['CH4', 'NOx', 'VOC', 'HC', 'N2O']:
    print(specie, parameters['best_fit'][specie]/normalisation[specie])


# In[ ]:


beta_hc_sum = 0

for species in hc_species:
    beta_hc = (
        p[3] * (
            (hc_eesc[species][264] - hc_eesc[species][100])/(total_eesc[264]-total_eesc[100])
        )
    )
    print(species, beta_hc)
    beta_hc_sum = beta_hc_sum + beta_hc
print(beta_hc_sum)


# In[ ]:


lifetime_scaling['best_fit'] = alpha_scaling_exp(
    input,
    baseline,
    normalisation,
    parameters['best_fit'],
)


# In[ ]:


pl.plot(np.arange(1750, 2101), lifetime_scaling['best_fit'])


# In[ ]:


lifetime_scaling['best_fit'][0]


# In[ ]:


lifetime_scaling['best_fit'][0] * parameters['best_fit']['base']


# In[ ]:


pl.plot(np.arange(1750, 2101), lifetime_scaling['best_fit'] * parameters['best_fit']['base'], label='best_fit')
pl.legend()
pl.ylabel('CH4 chemical lifetime (yr)')
#pl.savefig('../plots/ch4_chemical_lifetime_best_fit.pdf')


# In[ ]:


conc_ch4['best_fit'] = np.zeros(351)
gas_boxes = 0
airborne_emissions = 0
for i in range(351):
    conc_ch4['best_fit'][i], gas_boxes, airborne_emissions = one_box(
        emis_ch4[i],
        gas_boxes,
        airborne_emissions,
        burden_per_emission,
        parameters['best_fit']['base'],
        lifetime_scaling['best_fit'][i],
        partition_fraction,
        pre_industrial_concentration,
        soil_lifetime=np.inf,
        timestep=1,
        natural_emissions_adjustment=natural_emissions_adjustment,
    )


# ### Compare the SSP3-7.0 fit to other SSPs
#
# should do something with the temperature projections here

# In[ ]:


emis_ch4_ssps = {}

for ssp in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']:
    emis_ch4_ssps[ssp] = df_emis.loc[
        (df_emis['Scenario']==ssp) & (df_emis['Variable'].str.endswith('CH4')) &
        (df_emis['Region']=='World'), '1750':'2100'
    ].interpolate(axis=1).values.squeeze()


# In[ ]:


for ssp in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']:
    conc_ch4[ssp] = np.zeros(351)
    gas_boxes = 0
    airborne_emissions = 0
    for i in range(351):
        conc_ch4[ssp][i], gas_boxes, airborne_emissions = one_box(
            emis_ch4_ssps[ssp][i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            parameters['best_fit']['base'],
            lifetime_scaling['best_fit'][i],
            partition_fraction,
            pre_industrial_concentration,
            soil_lifetime=np.inf,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )


# ### Four panel plot

# In[ ]:


ar6_colors = {
    'ssp119': '#00a9cf',
    'ssp126': '#003466',
    'ssp245': '#f69320',
    'ssp370': '#df0000',
    'ssp434': '#2274ae',
    'ssp460': '#b0724e',
    'ssp534-over': '#92397a',
    'ssp585': '#980002'
}


# In[ ]:


fig, ax = pl.subplots(1, 3, figsize=(15, 4.5))
for model in models:
    ax[0].plot(np.arange(1750, 2101), lifetime_scaling[model] * parameters[model]['base'], label=model)
ax[0].plot(np.arange(1750, 2101), lifetime_scaling['best_fit'] * parameters[model]['base'], color='0.5', label='Best fit')
#ax[0].legend(loc='upper left', bbox_to_anchor=[0, 0.9], frameon=False)
ax[0].set_xlim(1750, 2100)
ax[0].set_ylabel('yr')
ax[0].set_title('(a) CH$_4$ lifetime SSP3-7.0')

for model in models:
    ax[1].plot(np.arange(1750, 2101), conc_ch4[model], label=model)
ax[1].plot(np.arange(1750, 2101), conc_ch4['best_fit'], color='0.5', label='Best fit')
ax[1].plot(np.arange(1750, 2101), input['CH4'], color='k', label='observations +\nMAGICC6')
ax[1].set_ylabel('ppb')
ax[1].set_xlim(1750, 2100)
ax[1].legend(frameon=False)
ax[1].set_title('(b) CH$_4$ concentration SSP3-7.0')

# ax[1,0].plot(np.arange(1750, 2101), conc_ch4['best_fit'], color='0.5', label='Best fit')
# ax[1,0].plot(np.arange(1750, 2101), input['CH4'], color='k', label='observations + MAGICC6')
# ax[1,0].set_ylabel('ppb')
# ax[1,0].set_xlim(1750, 2100)
# ax[1,0].set_title('(c) CH$_4$ concentration, best lifetime coefficient fit')
# ax[1,0].legend(frameon=False)

for ssp in ['ssp119', 'ssp126', 'ssp434', 'ssp534-over', 'ssp245', 'ssp460', 'ssp370',  'ssp585']:
    ax[2].plot(np.arange(1750, 2101), conc_ch4[ssp], label=ssp, color=ar6_colors[ssp])
ax[2].set_ylabel('ppb')
ax[2].set_title('(c) Best fit CH$_4$ projections')
ax[2].set_xlim(1750, 2100)
ax[2].legend(frameon=False)

fig.tight_layout()
# pl.savefig('../plots/methane_calibrations.png')
# pl.savefig('../plots/methane_calibrations.pdf')


# In[ ]:


# these are the feedback values per ppb / per Mt that go into FaIR
out = np.empty((1,7))
out[0,0] = lifetime_scaling['best_fit'][0] * parameters['best_fit']['base']
for i, specie in enumerate(['CH4', 'NOx', 'VOC', 'HC', 'N2O']):
    out[0,i+1] = parameters['best_fit'][specie]/normalisation[specie]
out[0,6] = parameters['best_fit']['temp']

df = pd.DataFrame(out, columns = ['base', 'CH4', 'NOx', 'VOC', 'HC', 'N2O', 'temp'], index = ['historical_best'])
df


# In[ ]:


df.to_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/CH4_lifetime.csv')
