# fair-calibrate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7112539.svg)](https://doi.org/10.5281/zenodo.7112539)

Multiple strategies to calibrate the FaIR model.

## installation

### requirements
- [`pixi`](https://pixi.sh) (installs Python, R, cmake and all packages in one locked environment; no separate `conda`, `python` or `R` install is needed)
- a C/C++ compiler for building the `FKF` and `EBM` R packages (Xcode command line tools on macOS, `gcc` on Linux)

### set up the environment for Python and R

Install `pixi` (see https://pixi.sh), then from the top directory of this repository run

```
pixi install
pixi run install-ebm
```

The first command creates the environment from `pixi.toml` and `pixi.lock`: `python` 3.11, `R>=4.4`, `cmake`, the R packages available on conda-forge (`expm`, `nloptr`, `numDeriv`) and the Python packages, including `fair` and this repository (editable). The second installs the two R packages that conda-forge does not provide: `FKF` from CRAN and Donald Cummins' [`EBM`](https://github.com/donaldcummins/EBM) v1.1.0 from GitHub. It only needs to be run once per environment.

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
```

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
1. I get different results from the 3-layer model calibration between using pre-compiled R binary for for Mac compared to building the R binary from source on CentOS7; both using R-4.1.1, and again using the Arc4 HPC. The Arc4 results are used. A future **TODO** would be to switch to ``py-bobyqa`` which is the optimizer used in the R code, and remove dependence on R, which *may* improve performace.
2. Related to above, scipy's multivariate normal and sparse matrix algebra routines seem fragile, and change between scipy versions (1.8, 1.9, 1.10). If anyone trying to reproduce this runs into "positive semidefinite" errors, raise an issue.

## Documentation
More details on each calibration version are in the [Wiki](https://github.com/chrisroadmap/fair-calibrate/wiki).

It is critical that each calibration version and calibration set is well documented, as they may be used by others: often, differences in the responses in climate emulators are more a function of calibration than of model structural differences (I don't have a single good reference to prove this yet, but trust me). New calibrations will not be accepted without a Wiki entry.
