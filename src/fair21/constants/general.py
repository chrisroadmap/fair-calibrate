"""
General constants
"""

import numpy as np

DOUBLING_TIME_1PCT = np.log(2)/np.log(1.01)  # about 69.7
EARTH_RADIUS = 6371000 # m
M_ATMOS = 5.1352e18 # mass of atmosphere, kg
NBOX = 3  # hard code for now, to do is a truly flexible model
SECONDS_PER_YEAR = 60 * 60 * 24 * 365.24219 # Length of tropical year
    # https://en.wikipedia.org/wiki/Tropical_year

SCENARIO_AXIS = 0
SPECIES_AXIS = 1
TIME_AXIS = 2
GAS_BOX_AXIS = 3
