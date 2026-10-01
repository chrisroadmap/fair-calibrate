"""The number of ocean layers in the energy balance model, shared by the pipeline.

The calibration (``input/calibration/02``, ``03``), sampling and constraining
scripts all follow ``N_LAYERS`` (2 or 3) from ``parameters.py``, so changing the
layer count is a one-line edit there that is committed with the calibration.
It is deliberately not an environment variable: a machine-local ``.env`` could
silently contradict the committed value. File names carry the layer count
(``climate_response_ebm2.csv``), as the calibration output already did
(``4xCO2_cummins_ebm2_cmip6.csv``).
"""

import numpy as np
from fair.interface import fill

from fair_calibrate import parameters
from fair_calibrate.cummins_ebm import SUPPORTED_LAYERS


def get_n_layers():
    """Return ``N_LAYERS`` from ``fair_calibrate.parameters``, checked."""
    n_layers = parameters.N_LAYERS  # read at call time, so tests can override it
    if n_layers not in SUPPORTED_LAYERS:
        raise SystemExit(
            f"N_LAYERS in parameters.py must be one of {SUPPORTED_LAYERS}, "
            f"got {n_layers}"
        )
    return n_layers


def calibration_file(n_layers):
    """Calibration table written by ``calibration/02_calibrate_cummins.py``."""
    return f"4xCO2_cummins_ebm{n_layers}_cmip6.csv"


def climate_response_file(n_layers):
    """Prior climate-response sample written by ``sampling/01_...``."""
    return f"climate_response_ebm{n_layers}.csv"


def heat_capacity_columns(n_layers):
    """Column names of the layer heat capacities in the prior sample."""
    return [f"c{i}" for i in range(1, n_layers + 1)]


def heat_transfer_columns(n_layers):
    """Column names of the layer heat exchange coefficients in the prior sample."""
    return [f"kappa{i}" for i in range(1, n_layers + 1)]


def climate_response_columns(n_layers):
    """Columns of the prior sample, in the order the sampler produces them."""
    return (
        ["gamma"]
        + heat_capacity_columns(n_layers)
        + heat_transfer_columns(n_layers)
        + ["epsilon", "sigma_eta", "sigma_xi", "F_4xCO2"]
    )


def climate_response_renames(n_layers):
    """Map prior-sample columns to FaIR's climate config names (for the dump)."""
    renames = {"gamma": "gamma_autocorrelation"}
    for i in range(n_layers):
        renames[f"c{i + 1}"] = f"ocean_heat_capacity[{i}]"
    for i in range(n_layers):
        renames[f"kappa{i + 1}"] = f"ocean_heat_transfer[{i}]"
    renames.update(
        {
            "epsilon": "deep_ocean_efficacy",
            "sigma_eta": "sigma_eta",
            "sigma_xi": "sigma_xi",
            "F_4xCO2": "forcing_4co2",
        }
    )
    return renames


def layer_config(df_cr, rows, n_layers):
    """Per-layer entries of a run configuration, from the prior sample.

    ``rows`` is anything ``DataFrame.loc`` accepts as a row selector. The result
    has ``n_layers`` plus ``c1..cN`` and ``kappa1..kappaN``, which
    :func:`fill_layers` reads back.
    """
    config = {"n_layers": n_layers}
    for column in heat_capacity_columns(n_layers) + heat_transfer_columns(n_layers):
        config[column] = df_cr.loc[rows, column].values
    return config


def fill_layers(f, cfg):
    """Fill the layer heat capacities and exchange coefficients of a FAIR instance.

    ``f`` must have been built with ``FAIR(n_layers=cfg["n_layers"])``.
    """
    n_layers = cfg["n_layers"]
    fill(
        f.climate_configs["ocean_heat_capacity"],
        np.array([cfg[column] for column in heat_capacity_columns(n_layers)]).T,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        np.array([cfg[column] for column in heat_transfer_columns(n_layers)]).T,
    )
