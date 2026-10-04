"""TEMPORARY override of the carbon cycle priors from fair-carbon-feedbacks.

Set ``CARBON_CYCLE_FILE`` (in ``.env``) to a CSV from fair-carbon-feedbacks with the
IIRF parameters and the four-box partition fractions and unperturbed lifetimes of CO2.
Row ``i`` of the file goes with row ``i`` of every other prior. Where it is set it
replaces the sample from ``input/sampling/04`` in sampling/10, constraining/02, 05
and 07. Unset, nothing here does anything and the pipeline is unchanged.

FaIR's ``override_defaults`` takes one index per column header, so it cannot set
per-box parameters: constraining/05 writes them to a separate file
(:data:`BOXES_FILE`) and constraining/07 applies them with :func:`set_boxes`.
"""

import os

import numpy as np
import pandas as pd

IIRF_COLUMNS = ["iirf_0", "iirf_airborne", "iirf_uptake", "iirf_temperature"]
PARTITION_COLUMNS = [f"partition_fraction{i}" for i in range(4)]
LIFETIME_COLUMNS = [f"unperturbed_lifetime{i}" for i in range(4)]
BOXES_FILE = "calibrated_constrained_carbon_boxes.csv"


def load_override(min_rows):
    """Return the override table, or None when ``CARBON_CYCLE_FILE`` is not set."""
    path = os.getenv("CARBON_CYCLE_FILE")
    if not path:
        return None
    print(f"OVERRIDING carbon cycle priors (04) with {path}")
    df = pd.read_csv(path)
    if df.shape[1] == 1:  # European export: semicolon delimiter
        df = pd.read_csv(path, sep=";")
    required = IIRF_COLUMNS + PARTITION_COLUMNS + LIFETIME_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CARBON_CYCLE_FILE is missing columns: {missing}")
    if len(df) < min_rows:
        raise SystemExit(
            f"CARBON_CYCLE_FILE has {len(df)} rows, {min_rows} (PRIOR_SAMPLES) needed"
        )
    if "member" in df.columns and not np.array_equal(
        df["member"].values, np.arange(len(df))
    ):
        raise SystemExit("CARBON_CYCLE_FILE: 'member' is not 0..n-1 in row order")
    return df.reset_index(drop=True)


def carbon_cycle_config(df, rows):
    """IIRF and box entries of a run configuration for the given row labels."""
    config = {name: df.loc[rows, name].values for name in IIRF_COLUMNS}
    config["partition_fraction"] = df.loc[rows, PARTITION_COLUMNS].values
    config["unperturbed_lifetime_co2"] = df.loc[rows, LIFETIME_COLUMNS].values
    return config


def set_boxes(f, partition_fraction, unperturbed_lifetime):
    """Set the four CO2 boxes of a FAIR instance, one row per config."""
    co2 = dict(specie="CO2")
    f.species_configs["partition_fraction"].loc[co2] = partition_fraction
    f.species_configs["unperturbed_lifetime"].loc[co2] = unperturbed_lifetime


def dump_columns(df):
    """IIRF columns named as ``calibrated_constrained_parameters.csv`` expects."""
    return df[IIRF_COLUMNS].rename(columns={c: f"{c}[CO2]" for c in IIRF_COLUMNS})
