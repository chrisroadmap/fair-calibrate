#!/usr/bin/env python
# coding: utf-8

# # Make emissions binary file
#
# SSPs from RCMIP

# In[ ]:


import os
from fair import FAIR
from fair.interface import initialise, fill
from fair.io import read_properties

from dotenv import load_dotenv
from fair import __version__


# Get environment variables
load_dotenv()

cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__


# In[ ]:


scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']


# In[ ]:


species, properties = read_properties()
species.remove("Contrails")
species.remove("NOx aviation")


# In[ ]:


f = FAIR(ch4_method='thornhill2021')
f.define_time(1750, 2500, 1)
f.define_configs(['unspecified'])
f.define_scenarios(scenarios)
f.define_species(species, properties)


# In[ ]:


f.allocate()
f.fill_from_rcmip()


# In[ ]:


f.emissions


# In[ ]:


os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/')


# In[ ]:


f.emissions.to_netcdf(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/ssp_emissions_1750-2500.nc')


# In[ ]:
