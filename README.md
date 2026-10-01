# fair-calibrate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7112539.svg)](https://doi.org/10.5281/zenodo.7112539)

Multiple strategies to calibrate the FaIR model.

## installation

### requirements
- [`pixi`](https://pixi.sh) (installs Python, R, cmake and all packages in one locked environment; no separate `conda`, `python` or `R` install is needed)
- optional, only for the R reference implementation: a C/C++ compiler for building the `FKF` and `EBM` R packages (Xcode command line tools on macOS, `gcc` on Linux)

### set up the environment

Install `pixi` (see https://pixi.sh), then from the top directory of this repository run

```
pixi install
```

This creates the environment from `pixi.toml` and `pixi.lock`: `python` 3.11, the Python packages (including `fair`, `nlopt` and this repository, editable), and `R>=4.4` with `cmake`.

The Cummins calibration (`input/calibration/02_calibrate_cummins.py`, two or three layers) is pure Python and does not need R. R is kept for the reference implementation in `alternatives/cummins/` and for regenerating the R test values in `tests/`. To use it, install the two R packages that conda-forge does not provide, `FKF` from CRAN and Donald Cummins' [`EBM`](https://github.com/donaldcummins/EBM) v1.1.0 from GitHub, once per environment:

```
pixi run install-ebm
```

To work interactively, either prefix commands with `pixi run` (e.g. `pixi run python script.py`) or open a shell inside the environment with `pixi shell`.

Supported platforms are `osx-arm64` and `linux-64`. For another platform, add it to `platforms` in `pixi.toml`.

### if you need to add a new package or update the calibration version
Add the package with `pixi add <package>` (from conda-forge) or `pixi add --pypi <package>` (from PyPI). This updates `pixi.toml` and `pixi.lock`; commit both. Update the pinned `fair` version by editing the `fair` entry under `[pypi-dependencies]` in `pixi.toml`, then run `pixi install`.

Since the calibration label information is part of the package, the editable install picks up edits to `src/fair_calibrate/parameters.py` and `setup.py` automatically. If you change `setup.py` metadata such as the version, run `pixi reinstall` to refresh it.

## How to run

### Create an `.env` file in the top directory

The `.env` file contains machine-specific environment variables that should be changed in order to produce the calibration. These are FaIR version, calibration version, constraints context and number of samples.

```
# example .env
BATCH_SIZE=500               # how many scenarios to run in parallel
WORKERS=40                   # how many cores to use for parallel runs
FRONT_SERIAL=0               # for debugging, how many serial runs to do first
FRONT_PARALLEL=0             # after serial runs, how many parallel runs to test

PLOTS=True                   # Produce plots?
PROGRESS=False               # show progress bar? (good for interactive, bad
                             # on HPC batch jobs)
DATADIR=/path/to/datadir     # A local location to download large external
                             # datafiles (a cache directory)

GAMMA_MAX=10                 # optional: upper bound on gamma, the stochastic
                             # forcing autocorrelation rate (1/yr), in the
                             # three-layer fit. Default 10; "none" removes it.
DEEP_MIN_RATIO=1             # optional: lower bound on the deepest layer's heat
                             # capacity over the one above (C3/C2 for three
                             # layers, C2/C1 for two). Default 1; "none" removes it.
N_LAYERS=3                   # ocean layers in the Cummins model, 2 or 3. Sets
                             # which of 4xCO2_cummins_ebm2/3_cmip6.csv is written.
```

`WORKERS` also sets how many model/run fits `02_calibrate_cummins.py` runs in parallel. The fit is a long loop over small matrices, so it is pinned to one BLAS thread per worker.

Then, if necessary, edit the file in `src/fair_calibrate/parameters.py` and `setup.py` to point to the correct version and label.

The output will be produced in `output/`. No posterior data will be committed to Git owing to size, but the intention is that the full output data will be on Zenodo.

### To run the workflow
1. Create the `.env` file - see above
2. Set up the environment for python and R (see above)
3. Check the recipe inside the `run` bash script
4. `pixi run run`

During diagnosis and debugging, scripts can be run individually, but must be run from the directories in which they reside (5 subdirectories deep). If you do this, run them inside the environment (with `pixi run` or `pixi shell`, which works from any subdirectory of the repository).

Under the existing pattern -- which you are free to change in the `run` recipe -- scripts are automatically run in numerical order by the workflow if they are prefixed with a two digit number and an underscore, in this order:
- `calibration/`
- `sampling/`
- `constraining/`

### To produce a new calibration
1. Create your workflow scripts inside `input/` (copy an existing calibration to get started)
2. Set up the environment for python and R (see above)
3. Update your `.env` file
4. Check the recipe inside the `run` bash script
5. `pixi run run`
6. Check output. Ensure the performance metrics are documented and diagnostic plots look sensible.
7. If releasing a new calibration: update the relevant sections of the [Wiki](https://github.com/chrisroadmap/fair-calibrate/wiki)
8. `./create_zenodo_zip`
9. Upload to Zenodo

## Notes
1. The Cummins energy balance model calibration (two or three layers) is now fitted in Python (`src/fair_calibrate/cummins_ebm.py`), using NLopt's BOBYQA, the same library that R's `nloptr` wraps. Earlier versions used Donald Cummins' R package, and gave different results between a pre-compiled R binary on Mac and a source build on CentOS7 (both R-4.1.1) and again on the Arc4 HPC. The Python port reproduces the R fit to about 1e-6 for most runs (`pixi run test` checks it against stored R values), with two deliberate differences:
   - For about a fifth of the CMIP6 runs the exact likelihood keeps improving as `gamma` grows without limit (the white-noise limit, which annual data cannot distinguish from a large finite `gamma`). R's fits for these runs stop at `gamma` of 24 to 30, apparently because its matrix exponential loses accuracy there (the likelihood also gets worse for R's fits than for the unbounded optimum), not because the data say so. The Python fit bounds `gamma` at `GAMMA_MAX` (default 10), and the `gamma_at_bound` column of the output marks the runs that end on it.
   - The deepest layer is constrained to be at least as large as the one above it (`DEEP_MIN_RATIO`, so `C3 >= C2` for three layers). Without it, three-layer fits could collapse `C3` to almost nothing while the deep-ocean efficacy `epsilon` ran to 100 or more. The `deep_at_bound` column marks fits that sit on the constraint, meaning the data would prefer a deepest layer smaller than the one above.
   - Every run is fitted from two starting points. The better result is kept, and the `start_gap` column records how far apart the two fits were; a large gap, or only one start converging, flags a fit to check.
   - The `suspect` and `suspect_reasons` columns mark fits to look at before using them: `epsilon` outside 0.5 to 2.5, `C3/C2` on its bound, the two starts disagreeing, or only one start converging. `epsilon` is flagged, not bounded.
3. `N_LAYERS=2` fits the two-layer model and converts it (`03_convert-ebm3-to-impulse-response.py` reads `N_LAYERS` too), writing `4xCO2_cummins_ebm2_cmip6.csv` and `4xCO2_impulse_response_ebm2_cmip6.csv`. The later steps (`sampling/01_climate-response-sampling-ebm3.py`, `constraining/02_run-1pct.py` and `05_dump-calibration.py`, `sampling/06_forcing-uncertainty-ar6.py`, `sampling/10_run-fair-ssp-prior-ensemble-ebm3-intvar.py`) still read the three-layer files only, so a two-layer FaIR calibration needs those migrated.
2. Related to above, scipy's multivariate normal and sparse matrix algebra routines seem fragile, and change between scipy versions (1.8, 1.9, 1.10). If anyone trying to reproduce this runs into "positive semidefinite" errors, raise an issue.

## Documentation
More details on each calibration version are in the [Wiki](https://github.com/chrisroadmap/fair-calibrate/wiki).

It is critical that each calibration version and calibration set is well documented, as they may be used by others: often, differences in the responses in climate emulators are more a function of calibration than of model structural differences (I don't have a single good reference to prove this yet, but trust me). New calibrations will not be accepted without a Wiki entry.
