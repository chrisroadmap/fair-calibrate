"""
Default parameters relating to greenhouse gases
"""

import numpy as np

# ppt unless specified
pre_industrial_concentration = {
	"CO2" : 278.3, # ppm
}

partition_fractions = {
    "CO2": np.array([0.2173, 0.2240, 0.2824, 0.2763])
}
