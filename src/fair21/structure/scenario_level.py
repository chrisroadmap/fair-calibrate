from dataclasses import dataclass
import typing

from .top_level import Category, RunMode, AggregatedCategory
from .species_level import Species
from ..exceptions import IncompatibleConfigError

@dataclass
class Scenario():
    name: str
    list_of_species: typing.List[Species]

    def __post_init__(self):
        if not isinstance(self.list_of_species, list):
            raise TypeError('list_of_species argument passed to Scenario must be a list of Species')

        # check for Categories for which it makes no sense to duplicate
        running_total = {category: 0 for category in AggregatedCategory.NO_DUPLICATES_ALLOWED}
        major_ghgs_forward_mode = 0
        for species in self.list_of_species:
            if species.species_id.category in AggregatedCategory.NO_DUPLICATES_ALLOWED:
                running_total[species.species_id.category] = running_total[species.species_id.category] + 1
                if running_total[species.species_id.category] > 1:
                    raise DuplicationError(
                        f"The Scenario contains more than one instance of "
                        f"{species.species_id.category}. This is not valid."
                    )
            if species.species_id.category in [Category.CO2, Category.CH4, Category.N2O]:
                if species.species_id.run_mode in [RunMode.EMISSIONS, RunMode.CONCENTRATION]:
                    major_ghgs_forward_mode = major_ghgs_forward_mode + 1
        n_major_ghgs = running_total[Category.CO2] + running_total[Category.CH4] + running_total[Category.N2O]
        if 0 < n_major_ghgs < 3:
            if major_ghgs_forward_mode > 0: #TODO and emissions or concentration driven mode
                raise IncompatibleConfigError(
                    f"Either all of CO2, CH4 and N2O must be given in a Scenario, or "
                    f"none, unless those provided are forcing-driven. If you want to "
                    f"exclude the effect of one or two of these gases, consider setting "
                    f"emissions of these gases to zero or concentrations to pre-industrial."
                )
