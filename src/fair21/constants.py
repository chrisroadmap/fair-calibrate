"""
General constants
"""

import numpy as np

DOUBLING_TIME_1PCT = np.log(2)/np.log(1.01)  # about 69.7
EARTH_RADIUS = 6371000 # m
M_ATMOS = 5.1352e18 # mass of atmosphere, kg
SECONDS_PER_YEAR = 60 * 60 * 24 * 365.24219 # Length of tropical year
    # https://en.wikipedia.org/wiki/Tropical_year

TIME_AXIS = 0
SCENARIO_AXIS = 1
CONFIG_AXIS = 2
SPECIES_AXIS = 3
GAS_BOX_AXIS = 4
