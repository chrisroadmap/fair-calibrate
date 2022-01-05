# Exceptions

class DuplicatedSpeciesError(Exception):
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

class TimeNotIterableError(Exception):
    pass

class UndefinedSpeciesError(Exception):
    pass

class UnknownRunModeError(Exception):
    pass

class UnexpectedInputError(Exception):
    pass
