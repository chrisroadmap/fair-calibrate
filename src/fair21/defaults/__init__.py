"""
Module for defaults
"""

import copy

from .default_species_config import default_species_config

IIRF_HORIZON = 100

# need to be moved
def species_config_from_default(name, **kwargs):
    # not making a copy would mutate the base dict: we don't want this
    config = copy.copy(default_species_config[name.lower()])
    for key, value in kwargs.items():
        setattr(config, key, value)
    config.__post_init__()  # ensures checks are run
    return config
