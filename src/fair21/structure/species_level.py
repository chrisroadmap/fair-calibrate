from dataclasses import dataclass

import numpy as np

from .top_level import SpeciesID, RunMode

# scenario level
@dataclass
class Species():
    """Defines the contained data within a Species.

    Attributes
    ----------
    species_id : SpeciesID
        Previously-defined SpeciesID that defines what we are associating the
        driving data with.
    emissions : :obj:`np.ndarray`, optional
        Time series of emissions.
    concentration : :obj:`np.ndarray`, optional
        Time series of atmospheric concentration for greenhouse gases.
    forcing : :obj:`np.ndarray`, optional
        Time series of effective radiative forcing (W m-2).

    Raises
    ------
    MissingInputError
        if data required is not provided.
    """
    species_id: SpeciesID
    emissions: np.ndarray=None
    concentration: np.ndarray=None
    forcing: np.ndarray=None

    def __post_init__(self):
        # Check if required data is present for the selected RunMode.
        if self.species_id.run_mode == RunMode.EMISSIONS and self.emissions is None:
            raise MissingInputError(f"for {self.species_id.name} run in emissions mode, emissions must be specified")
        if self.species_id.run_mode == RunMode.CONCENTRATION and self.concentration is None:
            raise MissingInputError(f"for {self.species_id.name} run in concentration mode, concentration must be specified")
        if self.species_id.run_mode == RunMode.FORCING and self.forcing is None:
            raise MissingInputError(f"for {self.species_id.name} run in forcing mode, forcing must be specified")
