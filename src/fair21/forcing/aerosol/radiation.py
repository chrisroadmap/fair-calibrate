import numpy as np

from ...defaults.aerosol.radiation import radiative_efficiency

def linear(
    emissions,
    pre_industrial_emissions,
    radiative_efficiency=radiative_efficiency["AR6"]
):
    """
    Calculate effective radiative forcing from aerosol-radiation interactions.

    Inputs
    ------
    emissions : dict of `np.ndarray` or float
        input emissions
    pre_industrial_emissions : dict of float
        pre-industrial emissions
    radiative_efficiency : dict of float
        radiative efficiency of each species.

    Returns
    -------
    erf_ari : dict
        ERF from aerosol radiation interactions from each component.
    """

    effective_radiative_forcing = {}

    species = ['Sulfur', 'BC', 'OC', 'NH3']
    for specie in species:
        effective_radiative_forcing['Aerosol|Aerosol-radiation interactions|{}'.format(specie)] = (
            radiative_efficiency[specie] *
            (emissions[specie] - pre_industrial_emissions[specie])
        )

    # needs to be in a constants module
    for ispecie, specie in enumerate(species):
        if ispecie==0:
            effective_radiative_forcing['Aerosol|Aerosol-radiation interactions'] = np.zeros_like(emissions[specie])
        effective_radiative_forcing['Aerosol|Aerosol-radiation interactions'] = (
            effective_radiative_forcing['Aerosol|Aerosol-radiation interactions'] +
            effective_radiative_forcing['Aerosol|Aerosol-radiation interactions|{}'.format(specie)]
        )

    return effective_radiative_forcing
