import numpy as np


def fill(var, data, **kwargs):
    """Fills a `fair` variable instance.

    One could work directly with the `xarray` `DataArray`s, but this function
    includes additional error checking and validation, and is slightly less
    cumbersome than the `xarray` method of allocation.

    Parameters:
        var : `FAIR` variable attribute to fill
            for example fair.climate_configs["ocean_heat_capacity"]
        data : `np.ndarray` like
            data to fill the variable with
        **kwargs : `str`s
            the dimensions represeted in the `data`

    Raises:
        ValueError :
            if a `kwarg` provided doesn't correspond to a dimension name in `var`
        ValueError :
            if `data` cannot be broadcast to `var`
    """

    # does variable exist?
    for kwarg in kwargs:
        if kwarg not in var.coords:
            raise ValueError(
                f"{kwarg} is not a coordinate of {var.name}. Valid coordinates are {var.coords._names}"
            )

    #    # are arrays broadcastable? this check isn't strictly needed as will fail loudly but is nicer than the default exception
    #    data_array = np.atleast_1d(np.asarray(data))
    #    var_shape = var.shape
    #    data_shape = data_array.shape
    #    if not all((m == n) or (m == 1) or (n == 1) for m, n in zip(var_shape[::-1], data_shape[::-1])):
    #        # this check thanks to https://stackoverflow.com/questions/47243451/checking-if-two-arrays-are-broadcastable-in-python
    #        raise ValueError(f"trying to fill data of shape {data_shape} to variable of shape {var_shape}")
    var.loc[kwargs] = data


def initialise(var, value, **kwargs):
    """Fills a `fair` variable instance with `value` in first timebound

    Otherwise identical to `fill`.

    Parameters:
        var : `FAIR` variable attribute to fill
            for example fair.climate_configs["ocean_heat_capacity"]
        value : `np.ndarray` like
            value to fill the first timebound with
        **kwargs : `str`s
            indices of the dimensions represeted in the `data`
    """

    # check value is a scalar?
    fill(var[0, ...], value, **kwargs)
