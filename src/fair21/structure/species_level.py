from dataclasses import dataclass

import numpy as np

from .top_level import SpeciesID, RunMode

# scenario level
@dataclass
class Species():
    species_id: SpeciesID
    emissions: np.ndarray=None
    concentration: np.ndarray=None
    forcing: np.ndarray=None
    #run_mode: field() - define at top level

    def __post_init__(self):
        # 1. Input validation
        if self.species_id.run_mode == RunMode.EMISSIONS and self.emissions is None:
            raise MissingInputError(f"for {self.species_id.name} run in emissions mode, emissions must be specified")
        if self.species_id.run_mode == RunMode.CONCENTRATION and self.concentration is None:
            raise MissingInputError(f"for {self.species_id.name} run in concentration mode, concentration must be specified")
        if self.species_id.run_mode == RunMode.FORCING and self.forcing is None:
            raise MissingInputError(f"for {self.species_id.name} run in forcing mode, forcing must be specified")
