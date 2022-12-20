# fair-calibrate
Multiple strategies to calibrate the FaIR model.

## installation

### requirements
- `anaconda` for `python3`
- `python` 3.6+
- for the Cummins calibration, a reasonably modern version of `R` (4.1.1 used here)

### python and jupyter notebooks
```
conda env create -f environment.yml
conda activate fair-calibrate
nbstripout --install
```

If you get module import errors running any of the notebooks, it's likely that your local environment is not up to date:
```
cd fair-calibrate  # path to wherever your local copy is

# get latest version
git fetch
git pull

# update environment
conda activate fair2.1-calibrate
conda env update -f environment.yml --prune
```

## How to run

The `.env` file contains environment variables that should be changed in order to produce the calibration. These are FaIR version, calibration version, constraints context and number of samples.

The output will be produced in `output/fair-X.X.X/vY.Y.Y/context` where X is the FaIR version, Y is the calibration version and context is the set of constraints used (e.g. AR6, 2022 observations, etc.).


To reproduce:

1. Set the environment variables.
2. Run the notebooks inside `notebooks/calibration`.
3. Run the R scripts inside `r_scripts`.
4. Run the notebooks inside `notebooks/sampling`.
5. Run the notebooks inside `notebooks/constraining`.

### the R scripts

Open the R console and set the working directory to the `r_scripts` directory of your local repository.

```
source("setup.r")
source("calibrate_cummins_3layer.r")
source("calibrate_cummins_2layer.r")
source("calibrate_cummins_3layer_longrunmip.r")
```

## Calibrations:

### v1.0
- 1.5 million prior ensemble
- **Carbon cycle** uses the parameters from Leach et al. 2021 calibrated for FaIR 2.0.0 using 11 C4MIP models.
- **Aerosol cloud interactions** depend on SO2, BC and OC, using calibrations from Smith et al. 2021 (developed for AR6) using 11 RFMIP and AerChemMIP models.
- **Aerosol radiation interactions** use prior values from AR6 Ch6, with a factor of two uncertainty for each species. ....
- ...

## Constraining contexts

### AR6
- Emissions and concentrations from RCMIP (i.e. based on CMIP6)
- Temperature from AR6 WG1 (1850-2020, mean of 4 datasets), constrained using ssp245 projections beyond 2014.
- Ocean heat content from AR6 WG1 (1971-2018), linear.
- two step constraining procedure used: first RMSE of less than 0.16K, then 6-variable distribution fitting.
- 1001-member posterior (deliberately chosen).

## acknowledgements and contributions

Should I ever write a calibration paper, these people should be on it.

1. Nick Leach and Stuart Jenkins for the original FaIR v2.0, which hopefully isn't too mangled or complicated by this attempt.
2. Hege-Beate Fredriksen for crunching the CMIP6 4xCO2 data from many more models
3. Donald Cummins for the three-layer model tuning algorithm.
4. Bill Collins for assistance with chemistry emulations and correspondence to AerChemMIP results.
5. Zeb Nicholls for advice on constraining to assessed ranges (e.g. AR6)
