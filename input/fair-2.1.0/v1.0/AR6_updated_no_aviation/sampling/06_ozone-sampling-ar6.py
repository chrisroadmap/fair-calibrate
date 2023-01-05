#!/usr/bin/env python
# coding: utf-8

# # Ozone calibration
#
# Use the AR6 per-species ozone calibrations, from Chapter 7 of IPCC AR6. These are not generated fresh here.

# In[ ]:


import os
import pandas as pd
import numpy as np
import scipy.stats

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


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)


# In[ ]:


scalings = scipy.stats.norm.rvs(
    loc=np.array(  [0.000175, 0.000710,-0.000125, 0.000155, 0.000329, 0.001797]),
    scale=np.array([0.000062, 0.000471, 0.000113, 0.000131, 0.000328, 0.000983])/NINETY_TO_ONESIGMA,
    size=(samples, 6),
    random_state=52
)


# In[ ]:


df = pd.DataFrame(scalings, columns=['CH4','N2O','Equivalent effective stratospheric chlorine','CO','VOC','NOx'])


# In[ ]:


os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/', exist_ok=True)
df.to_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv', index=False)


# In[ ]:
