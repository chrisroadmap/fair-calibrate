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

### Create an `.env` file in the top directory

The `.env` file contains environment variables that should be changed in order to produce the calibration. These are FaIR version, calibration version, constraints context and number of samples.

```
# example .env
CALIBRATION_VERSION=1.0
FAIR_VERSION=2.1.0
CONSTRAINTS=ar6
PRIOR_SAMPLES=1500000
POSTERIOR_SAMPLES=1001
CONSTRAINT_SET=AR6_updated
```

The output will be produced in `output/fair-X.X.X/vY.Y.Y/` where X is the FaIR version and Y is the calibration version. Multiple constraint philosphies can be applied for the same set of calibrations; these are in the posteriors/CONSTRAINTS subdirectory where CONSTRAINTS is a named set of constraints used (e.g. AR6, 2022 observations, etc.). The posteriors are the only data output that will be committed to Git, but the intention is that the full workflow will be on Zenodo.

### To run the workflow

actually this isn't quite right - the R scripts come between notebooks 2 and 3 in the calibration I think. TODO: make an automated workflow.

1. Create the `.env` file.
2. Run the notebooks inside `notebooks/calibration`.
3. Run the R scripts inside `r_scripts`.
4. Run the notebooks inside `notebooks/sampling`.
5. Run the notebooks inside `notebooks/constraining`.

#### Notebooks (steps 2, 3, 5)

From command line, make sure you are inside the environment, then launch Jupyter

```
conda activate fair-calibrate
jupyter notebook
```

### the R scripts

Open the R console (either the GUI or from the command line) and set the working directory to the `r_scripts` directory of your local repository.

```
setwd("r_scripts")
source("setup.r")
source("calibrate_cummins_3layer.r")
source("calibrate_cummins_2layer.r")
source("calibrate_cummins_3layer_longrunmip.r")
```

Note: I get different results from the 3-layer model calibration between using pre-compiled R binary for for Mac compared to building the R binary from source on CentOS7; both using R-4.1.1. Since the Linux OS is well out of date (Leeds IT: get your act together... again) and the Mac-derived results look way more sensible, the Mac results are the ones that should be used. A future **TODO** would be to switch to ``py-bobyqa`` which is the optimizer used in the R code, and remove dependence on R.

## Documentation:

It is critical that each calibration version and calibration set is well documented, as they may be used by others: often, differences in the responses in climate emulators are more a function of calibration than of model structural differences (we don't have a single good reference to prove this yet, but trust us).

Maybe a TODO: move to READTHEDOCS or a wiki.

### Calibration versions

#### v1.0
- 1.5 million prior ensemble
- **Climate response** calibrated on 49 abrupt-4xCO2 experiments from CMIP6 and sampled using correlated kernel density estimates
- **Methane lifetime** calibrated on 4 AerChemMIP experiments for 1850 and 2014 (Thornhill et al. 2021a, 2021b). Unlike other variables which are sampled around some prior uncertainty, only the best estimate historical calibration is used.
- **Carbon cycle** uses the parameters from Leach et al. 2021 calibrated for FaIR 2.0.0 using 11 C4MIP models.
- **Aerosol cloud interactions** depend on SO2, BC and OC, using calibrations from Smith et al. 2021a (developed for AR6) using 11 RFMIP and AerChemMIP models, with a prior of -2 to 0 W/m2.
- **Aerosol radiation interactions** use prior values from AR6 Ch6, with a factor of two uncertainty for each species and a prior in the range of -0.6 to 0.0.
- **Ozone** uses the same coefficeints as AR6 (Smith et al. 2021b).
- **Effective radaitive forcing uncertainty** follows the distributions in AR6.

Smith et al. 2021a: https://doi.org/10.1029/2020JD033622 |
Smith et al. 2021b: https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FGD_Chapter07_SM.pdf |
Thornhill et al. 2021a: https://doi.org/10.5194/acp-21-853-2021 |
Thornhill et al. 2021b: https://doi.org/10.5194/acp-21-1105-2021

### Constraint sets:

#### AR6_updated
- 1001-member posterior (deliberately chosen).
- Emissions and concentrations from RCMIP (i.e. based on CMIP6)
- Temperature from AR6 WG1 (1850-2020, mean of 4 datasets), constrained using ssp245 projections beyond 2014.
- Ocean heat content from AR6 WG1 (1971-2018), linear.
- two step constraining procedure used: first RMSE of less than 0.16K, then 6-variable distribution fitting.
- Aerosol ERF, ERFari and ERFaci as in AR6 WG1

#### AR6_updated_no_aviation
- As `AR6_updated`, but with historical and future forcing from aviation set to zero. The historical is recalibrated to switch aviation off.
