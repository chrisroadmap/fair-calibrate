#!/usr/bin/env python
# coding: utf-8

# # Climate response calibrations
#
# The purpose here is to provide correlated calibrations to the climate response in CMIP6 models.
#
# We will apply a very naive model weighting to the 4xCO2 results. We won't downweight for similar models*, but we will only select one ensemble member from models that provide multiple runs (the ensemble member that I deem the most reliable).
#
# *maybe the same model at different resolution should be downweighted.

# In[ ]:


import numpy as np
import pandas as pd
import os
import scipy.stats
import matplotlib.pyplot as pl
from tqdm import tqdm
from dotenv import load_dotenv

from fair.energy_balance_model import EnergyBalanceModel
from fair import __version__


# Get environment variables
load_dotenv()

cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
samples = int(os.getenv("PRIOR_SAMPLES"))


# In[ ]:


# pl.rcParams['figure.figsize'] = (5.875, 5.875)
# pl.rcParams['font.size'] = 12
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


df = pd.read_csv(
    os.path.join(f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/4xCO2_cummins_ebm3_cmip6.csv")
)


# In[ ]:


pd.set_option('display.max_rows', 78)
df


# In[ ]:


models = df['model'].unique()
models


# In[ ]:


for model in models:
    print(model, df.loc[df['model']==model, 'run'].values)


# Judgement time:
# - GISS-E2-1-G 'r1i1p1f1'
# - GISS-E2-1-H 'r1i1p3f1'  less wacky
# - MRI-ESM2-0 'r1i1p1f1'
# - EC-Earth3 'r3i1p1f1'  less wacky
# - FIO-ESM-2-0  'r1i1p1f1'
# - CanESM5  'r1i1p2f1'
# - FGOALS-f3-L 'r1i1p1f1'
# - CNRM-ESM2-1 'r1i1p1f2'

# In[ ]:


n_models=len(models)


# In[ ]:


n_models


# In[ ]:


multi_runs = {
    'GISS-E2-1-G': 'r1i1p1f1',
    'GISS-E2-1-H': 'r1i1p3f1',
    'MRI-ESM2-0': 'r1i1p1f1',
    'EC-Earth3': 'r3i1p1f1',
    'FIO-ESM-2-0':  'r1i1p1f1',
    'CanESM5':  'r1i1p2f1',
    'FGOALS-f3-L': 'r1i1p1f1',
    'CNRM-ESM2-1': 'r1i1p1f2',
}

params = {}

params[r'$\gamma$'] = np.ones(n_models) * np.nan
params['$c_1$'] = np.ones(n_models) * np.nan
params['$c_2$'] = np.ones(n_models) * np.nan
params['$c_3$'] = np.ones(n_models) * np.nan
params[r'$\kappa_1$'] = np.ones(n_models) * np.nan
params[r'$\kappa_2$'] = np.ones(n_models) * np.nan
params[r'$\kappa_3$'] = np.ones(n_models) * np.nan
params[r'$\epsilon$'] = np.ones(n_models) * np.nan
params[r'$\sigma_{\eta}$'] = np.ones(n_models) * np.nan
params[r'$\sigma_{\xi}$'] = np.ones(n_models) * np.nan
params[r'$F_{4\times}$'] = np.ones(n_models) * np.nan

for im, model in enumerate(models):
    if model in multi_runs:
        condition = (df['model']==model) & (df['run']==multi_runs[model])
    else:
        condition = (df['model']==model)
    params[r'$\gamma$'][im] = df.loc[condition, 'gamma'].values[0]
    params['$c_1$'][im], params['$c_2$'][im], params['$c_3$'][im] = df.loc[condition, 'C1':'C3'].values.squeeze()
    params[r'$\kappa_1$'][im], params[r'$\kappa_2$'][im], params[r'$\kappa_3$'][im] = df.loc[condition, 'kappa1':'kappa3'].values.squeeze()
    params[r'$\epsilon$'][im] = df.loc[condition, 'epsilon'].values[0]
    params[r'$\sigma_{\eta}$'][im] = df.loc[condition, 'sigma_eta'].values[0]
    params[r'$\sigma_{\xi}$'][im] = df.loc[condition, 'sigma_xi'].values[0]
    params[r'$F_{4\times}$'][im] = df.loc[condition, 'F_4xCO2'].values[0]


# In[ ]:


params = pd.DataFrame(params)


# In[ ]:


params.corr()


# In[ ]:


# fig = pl.figure()
# pd.plotting.scatter_matrix(params);
# pl.suptitle('Distributions and correlations of CMIP6 calibrations')
# pl.tight_layout()
# pl.subplots_adjust(wspace=0, hspace=0)
#pl.savefig('../plots/ebm3_distributions.png')


# In[ ]:


samples


# In[ ]:


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

kde = scipy.stats.gaussian_kde(params.T)
ebm_sample = kde.resample(size=int(samples*4), seed=2181882)

# remove unphysical combinations
for col in range(10):
    ebm_sample[:,ebm_sample[col,:] <= 0] = np.nan
ebm_sample[:, ebm_sample[0,:] <= 0.8] = np.nan  # gamma
ebm_sample[:, ebm_sample[1,:] <= 2] = np.nan   # C1
ebm_sample[:, ebm_sample[2,:] <= ebm_sample[1,:]] = np.nan    # C2
ebm_sample[:, ebm_sample[3,:] <= ebm_sample[2,:]] = np.nan # C3
ebm_sample[:, ebm_sample[4,:] <= 0.3] = np.nan                # kappa1 = lambda

mask = np.all(np.isnan(ebm_sample), axis=0)
ebm_sample = ebm_sample[:,~mask]
ebm_sample_df=pd.DataFrame(
    data=ebm_sample[:,:samples].T, columns=['gamma','c1','c2','c3','kappa1','kappa2','kappa3','epsilon','sigma_eta','sigma_xi','F_4xCO2']
)
ebm_sample_df


# In[ ]:


assert len(ebm_sample_df) >= samples


# In[ ]:


pl.hist(ebm_sample_df['kappa1'])
np.percentile(ebm_sample_df['kappa1'], (0,5,16,50,84,95,100))


# In[ ]:


pl.hist(3.934/ebm_sample_df['kappa1'])
np.percentile(3.934/ebm_sample_df['kappa1'], (0,5,16,50,84,95,100))


# In[ ]:


# # Since we use the actual scaled 2xCO2

# ecs = np.zeros(samples)
# tcr = np.zeros(samples)
# for i in tqdm(range(samples)):
#     ebm = EnergyBalanceModel(
#         ocean_heat_capacity = np.array([ebm_sample_df.loc[i,'c1'],ebm_sample_df.loc[i,'c2'],ebm_sample_df.loc[i,'c3']]),
#         ocean_heat_transfer = np.array([ebm_sample_df.loc[i,'kappa1'],ebm_sample_df.loc[i,'kappa2'],ebm_sample_df.loc[i,'kappa3']]),
#         deep_ocean_efficacy = ebm_sample_df.loc[i,'epsilon'],
#         forcing_4co2 = ebm_sample_df.loc[i,'F_4xCO2']
#     )
#     ebm.emergent_parameters()
#     #ebm.emergent_parameters(forcing_2co2_4co2_ratio=0.47630403979167685)
#     ecs[i] = ebm.ecs
#     tcr[i] = ebm.tcr


# In[ ]:


# ebm_sample_df['ecs'] = ecs
# ebm_sample_df['tcr'] = tcr


# In[ ]:


# pl.hist(ecs, bins=np.arange(0,10,0.1));
# np.percentile(ecs, (5, 16, 50, 84, 95))


# In[ ]:


# pl.hist(tcr, bins=np.arange(0,6,0.1));
# np.percentile(tcr, (5, 16, 50, 84, 95))


# In[ ]:


# pl.scatter(ecs, tcr)
# pl.xlim(0,10)
# pl.ylim(0,6)


# In[ ]:


os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/', exist_ok=True)


# In[ ]:


ebm_sample_df.to_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/climate_response_ebm3.csv', index=False)


# In[ ]:


# preserve correlation structure of kappa1 and F4x, even though we don't use F4x in the ensemble
pl.hist(ebm_sample_df['F_4xCO2']/ebm_sample_df['F_4xCO2'].mean())
np.percentile(ebm_sample_df['F_4xCO2']/ebm_sample_df['F_4xCO2'].mean(), (5, 50, 95))


# In[ ]:


# around best
np.percentile(1+ 0.563*(ebm_sample_df['F_4xCO2'].mean() - ebm_sample_df['F_4xCO2'])/ebm_sample_df['F_4xCO2'].mean(), (5,50,95))


# In[ ]:


# what we do want to do is to scale the variability in 4xCO2 (correlated with the other EBM parameters)
# to feed into the effective radiative forcing scaling factor.
1 + 0.563*(ebm_sample_df['F_4xCO2'].mean() - ebm_sample_df['F_4xCO2'])/ebm_sample_df['F_4xCO2'].mean()


# In[ ]:
