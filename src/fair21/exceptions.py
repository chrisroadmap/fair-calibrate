# Exceptions

class DuplicationError(Exception):
    pass

class ExternalForcingError(Exception):
    pass

class IncompatibleConfigError(Exception):
    pass

class InvalidRunModeError(Exception):
    pass

class LifetimeError(Exception):
    pass

class MissingInputError(Exception):
    pass

class NonNumericInputError(Exception):
    pass

class PartitionFractionError(Exception):
    pass

class ScenarioLengthMismatchError(Exception):
    pass

class SpeciesMismatchError(Exception):
    def __init__(self, scenario, config):
        if len(scenario) != len(config):
            self.message = (
                f"There are a different number of Species provided to "
                f"Scenario.list_of_species and Config.species_configs. "
                f"Scenario.list_of_species has {len(scenario)} elements and "
                f"Config.species_configs has {len(config)}. It's possible that "
                f"you forgot to define all of the Species in one or the other "
                f"or that there are duplicates."
            )
        else:
            for ispec in range(len(scenario)):
                if scenario[ispec] != config[ispec]:
                    self.message = (
                        f"The list of Species provided to "
                        f"Scenario.list_of_species differs from that provided to "
                        f"Config.species_configs. The first difference is in position "
                        f"{ispec}. Here Scenario.list_of_species[{ispec}] = "
                        f"{scenario[ispec]}, whereas Config.species_configs[{ispec}] = "
                        f"{config[ispec]}."
                    )
    def __str__(self):
        return self.message


class TimeMismatchError(Exception):
    pass

class TimeNotIterableError(Exception):
    pass

class UndefinedSpeciesError(Exception):
    pass

class UnknownRunModeError(Exception):
    pass

class UnexpectedInputError(Exception):
    pass

class WrongArrayShapeError(Exception):
    pass
