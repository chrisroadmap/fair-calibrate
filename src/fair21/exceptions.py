# Exceptions

class ExternalForcingError(Exception):
    pass

class IncompatibleConfigError(Exception):
    pass

class InvalidRunModeError(Exception):
    pass

class MissingInputError(Exception):
    pass

class NonNumericInputError(Exception):
    pass

class ScenarioLengthMismatchError(Exception):
    pass

class TimeNotDefinedError(Exception):
    pass

class UndefinedSpeciesError(Exception):
    pass

class UnknownRunModeError(Exception):
    pass
