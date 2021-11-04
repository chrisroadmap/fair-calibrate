import numpy as np
import pandas as pd
import scipy.interpolate

from ..exceptions import ExternalForcingError

# TODO: weighted averages when downscaling in time

def forcing_from_input(data, time, column=None, **kwargs):
    """Creates an effective_radiative_forcing input from a DataFrame or numpy array.

    Parameters
    ----------
    data : str, DataFrame or ndarray
        Input forcing data.
        If ndarray it must be two columns, the first being time and the second being the
        ERF on these time steps. If DataFrame, then the time should be the index, with the ERF being either the
        sole data column, or the column to extract specified with the `column` parameter.
    time : ndarray
        Array of times to extract
    column : str or None, optional
        column to extract from `data` if a DataFrame is provided.

    Other parameters
    ----------------
    see `scipy.interpolate.interp1d`.

    Raises
    ------
    ExternalForcingError:
        if forcing cannot be determined from input data.

    Returns
    -------
    forcing : ndarray
        array of effective radiative forcing.
    """
    if isinstance(data, str):
        data = pd.read_csv(data, index_col=0)
    if isinstance(data, pd.DataFrame):
        base_time = data.index.values
        if column is not None:
            base_data = data.loc[:,column].values.squeeze()
        else:
            if data.shape[1] != 1:
                raise ExternalForcingError("If not providing a column name to the forcing DataFrame, then only one column other than the time index should be provided.")
            base_data = data.iloc[:,0].values.squeeze()
    elif isinstance(data, np.ndarray):
        if data.shape[1] != 2:
            raise ExternalForcingError("If specifying a numpy array of external forcing, a (n_timestep, 2) shape array is required where the first column is time and the second column is forcing. The shape of the array you provided is {}.".format(data.shape))
        base_time = data[:,0]
        base_data = data[:,1]
    else:
        raise ExternalForcingError("Please provide a filepath (to a csv), a DataFrame or a numpy array")

    interp_func = scipy.interpolate.interp1d(base_time, base_data, **kwargs)
    data_out = interp_func(time)
    return data_out
