# fair-calibrate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7112539.svg)](https://doi.org/10.5281/zenodo.7112539)

Multiple strategies to calibrate the FaIR model.

## installation

### requirements
- `anaconda` for `python3`
- `python>=3.7`
- for the Cummins calibration, `R>=4.1.1` and `cmake>=3.2`

### set up environments for Python and R
```
conda env create -f environment.yml
conda activate fair-calibrate
cd r_scripts
R
> source("setup.r")
```

### if you need to add a new package
Edit the `environment.yml` file and add the package you require to the list. If it is conda installable and available under the list of channels provided, it should be picked up. To then update the environment, run
```
conda env update -f environment.yml --prune
```

## How to run

### Create an `.env` file in the top directory

The `.env` file contains environment variables that should be changed in order to produce the calibration. These are FaIR version, calibration version, constraints context and number of samples.

```
# example .env
CALIBRATION_VERSION=1.2
FAIR_VERSION=2.1.1
CONSTRAINT_SET=AR6_updated
PRIOR_SAMPLES=1500000        # how many prior samples to draw
POSTERIOR_SAMPLES=1001       # final posterior ensemble size
BATCH_SIZE=500               # how many scenarios to run in parallel
WORKERS=40                   # how many cores to use for parallel runs
FRONT_SERIAL=0               # for debugging, how many serial runs to do first
FRONT_PARALLEL=0             # after serial runs, how many parallel runs to test

PLOTS=True                   # Produce plots?
PROGRESS=False               # show progress bar? (good for interactive, bad
                             # on HPC batch jobs)
DATADIR=/path/to/data        # A location on disk to save downloaded data
```

The output will be produced in `output/fair-X.X.X/vY.Y.Y/Z/` where X is the FaIR version, Y is the calibration version and Z is the constraint set used. Multiple constraint philosphies can be applied for the same set of calibrations (e.g. AR6, 2022 observations, etc.). No posterior data will be committed to Git owing to size, but the intention is that the full output data will be on Zenodo.

### To run the workflow
1. Create the `.env` file - see above
2. Set up environments for python and R (see above)
3. Check the recipe inside the `run` bash script
4. `./run`

During diagnosis and debugging, scripts can be run individually, but must be run from the directories in which they reside (5 subdirectories deep). If you do this, activate your conda environment too (with `conda activate fair-calibrate`).

Under the existing pattern -- which you are free to change in the `run` recipe -- scripts are automatically run in numerical order by the workflow if they are prefixed with a two digit number and an underscore, in this order:
- `calibration/`
- `sampling/`
- `constraining/`

### To produce a new calibration
1. Create your workflow scripts inside `input/fair-X.X.X/vY.Y/Z` (copy an existing calibration to get started)
2. Set up environments for python and R (see above)
3. Update your `.env` file to point to the correct fair version (X.X.X), calibration version (vY.Y) and constraint set (Z)
4. Check the recipe inside the `run` bash script
5. `./run`
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
