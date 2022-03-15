# fair2.1-calibrate
FaIR version 2.1

What has been done?

- Re-calibration of FaIR v2.0 to updated Cummins calibration and inclusion of ocean heat content
- Adding back some of the emissions-driven relationships from FaIR v1.6
- changing the interface to a object oriented design

## installation

### requirements
- `anaconda` for `python3`
- `python` 3.7+ (enforced by `dataclass`)
- for the Cummins calibration, a reasonably modern version of `R` (4.1.1 used here)

### python and jupyter notebooks
```
conda env create -f environment.yml
conda activate fair2.1-calibrate
nbstripout --install
```

If you get module import errors running any of the notebooks, it's likely that your local environment is not up to date:
```
cd fair2.1-calibrate  # path to wherever your local copy is

# get latest version
git fetch
git pull

# update environment
conda activate fair2.1-calibrate
conda env update -f environment.yml --prune
```

TODO: put all of this into a config/make file.

### R scripts

Open the R console and set the working directory to the `r_scripts` directory of your local repository.

```
source("setup.r")
source("calibrate_cummins_3layer.r")
```

## reproduction

1. The `010_concatenate-hege-data-4xCO2.ipynb` notebook is run first.
2. Then, `r_scripts/calibrate_cummins_3layer.r` is run.
3. Finally, the remaining notebooks are run in numerical order.

## acknowledgements

1. Hege-Beate Fredriksen for crunching the CMIP6 data from all of the models
