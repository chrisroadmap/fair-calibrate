# fair-calibrate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7112539.svg)](https://doi.org/10.5281/zenodo.7112539)

Multiple strategies to calibrate the FaIR model.

## installation

### requirements
- `anaconda` for `python3`
- `python>=3.6`
- for the Cummins calibration, `R>=4.1.1` and `cmake>=3.2`

### set up environments for Python and R
```
conda env create -f environment.yml
conda activate fair-calibrate
cd r_scripts
R
> source("setup.r")
```

## How to run

### Create an `.env` file in the top directory

The `.env` file contains environment variables that should be changed in order to produce the calibration. These are FaIR version, calibration version, constraints context and number of samples.

```
# example .env
CALIBRATION_VERSION=1.1
FAIR_VERSION=2.1.0
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
```

The output will be produced in `output/fair-X.X.X/vY.Y.Y/Z/` where X is the FaIR version, Y is the calibration version and Z is the constraint set used. Multiple constraint philosphies can be applied for the same set of calibrations (e.g. AR6, 2022 observations, etc.). No posterior data will be committed to Git owing to size, but the intention is that the full output data will be on Zenodo.

### To run the workflow

1. Create the `.env` file - see above
2. Check the recipe inside the `run` bash script
3. ./run
4. ./create_zenodo_zip

Scripts can be run individually, but must be run from the directories in which they reside (5 subdirectories deep).

Scripts are automatically run in numerical order by the workflow if they are prefixed with a two digit number and an underscore, in this order:
- `calibration/`
- `sampling/`
- `constraining/`

## Notes
1. I get different results from the 3-layer model calibration between using pre-compiled R binary for for Mac compared to building the R binary from source on CentOS7; both using R-4.1.1, and again using the Arc4 HPC. The Arc4 results are used. A future **TODO** would be to switch to ``py-bobyqa`` which is the optimizer used in the R code, and remove dependence on R, which *may* improve performace.
2. Related to above, scipy's multivariate normal and sparse matrix algebra routines seem fragile, and change between scipy versions (1.8, 1.9, 1.10). If anyone trying to reproduce this runs into "positive semidefinite" errors, raise an issue.

## Documentation:

It is critical that each calibration version and calibration set is well documented, as they may be used by others: often, differences in the responses in climate emulators are more a function of calibration than of model structural differences (I don't have a single good reference to prove this yet, but trust me).

Maybe a TODO: move to READTHEDOCS or a wiki.

## Versioning strategy:

- If adding a new constraint set to an existing calibration mechanism and nothing else changes, this is a *micro* version (e.g. 1.0 to 1.0.1). There is no need to create new folders under this version; everything that applies to v1.0 would be valid for v1.0.1, v1.0.2 etc. Most updated will likely fall into this category.
- If an update or tweak to an existing calibration mechanism would change previously submitted results, this is a *minor* version. We would create a new folder for these inputs and results. v1.1 would differ from v1.0 for the same constraints.
- If a new calibration strategy is completely different and would break backward compatibility, this is a *major* version (e.g. 2.0).

### Calibration versions
Only major and minor versions need documenting here.

#### v1.1

As v1.0 with following fixes and improvements:
- **Aerosol cloud interactions** recalibrated to 13 (up from 11) CMIP6 models, and calibration code based on APRP fixed (part of the climateforcing package).
- **NOx emissions** from RCMIP have been corrected, where biomass burning emissions were in units of NO and agricultural and fossil emissions were in units of NO2, but no conversion was made in RCMIP. This affects historical calibrations for aerosol direct forcing, ozone forcing, and methane lifetime.
- **Methane lifetime** now reports the correct base lifetime for 1750, which is used in all projections.
- **Ozone forcing** calibration brought inside the code, rather than using AR6 results.

#### v1.0
- 1.5 million prior ensemble
- **Climate response** calibrated on 49 abrupt-4xCO2 experiments from CMIP6 and sampled using correlated kernel density estimates
- **Methane lifetime** calibrated on 4 AerChemMIP experiments for 1850 and 2014 (Thornhill et al. 2021a, 2021b). Unlike other variables which are sampled around some prior uncertainty, only the best estimate historical calibration is used.
- **Carbon cycle** uses the parameters from Leach et al. 2021 calibrated for FaIR 2.0.0 using 11 C4MIP models.
- **Aerosol cloud interactions** depend on SO2, BC and OC, using calibrations from Smith et al. 2021a (developed for AR6) using 11 RFMIP and AerChemMIP models, with a prior of -2 to 0 W/m2.
- **Aerosol radiation interactions** use prior values from AR6 Ch6, with a factor of two uncertainty for each species and a prior in the range of -0.6 to 0.0.
- **Ozone** uses the same coefficeints as AR6 (Smith et al. 2021b).
- **Effective radaitive forcing uncertainty** follows the distributions in AR6.

### Constraint sets:
Tag micro versions here along with descriptions.

#### GCP_2022 (v1.1.1)
Same as v1.0.2.

#### AR6_updated (v1.1.0)
Same constraints as v1.0.

#### GCP_2022 (v1.0.2)
Same as v1.0.1 except:
- land use forcing from cumulative CO2 emissions was not consistent with the AR6 assessment because of the change in CO2 emissions dataset and has been fixed.

#### GCP_2022 (v1.0.1)
Same as v1.0, except:
- CO2 FFI and AFOLU emissions are from Global Carbon Project (Friedlingstein et al. 2022), up to and including 2022 which is an estimate.
- SSP CO2 emissions are harmonized, with 2021 as the harmonization year. So 2022 is the first year in which scenarios may differ.

#### AR6_updated (v1.0)
- 1001-member posterior (deliberately chosen).
- Emissions and concentrations from RCMIP (i.e. based on CMIP6)
- Temperature from AR6 WG1 (1850-2020, mean of 4 datasets), constrained using ssp245 projections beyond 2014.
- ssp245 projections for 2081-2100.
- Ocean heat content from AR6 WG1 (1971-2018), linear.
- two step constraining procedure used: first RMSE of less than 0.16K, then 6-variable distribution fitting.
- Aerosol ERF, ERFari and ERFaci as in AR6 WG1

### References
- Friedlingstein et al. 2022: https://doi.org/10.5194/essd-14-4811-2022
- Leach et al. 2021: https://doi.org/10.5194/gmd-14-3007-2021
- Smith et al. 2021a: https://doi.org/10.1029/2020JD033622
- Smith et al. 2021b: https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FGD_Chapter07_SM.pdf
- Thornhill et al. 2021a: https://doi.org/10.5194/acp-21-853-2021
- Thornhill et al. 2021b: https://doi.org/10.5194/acp-21-1105-2021
