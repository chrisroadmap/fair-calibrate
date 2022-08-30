# fair2.1-calibrate
FaIR version 2.1, calibration to CMIP and constraining to AR6

What's new compared to v2.0 and v1.6:

- Re-calibration of FaIR v2.0 three-layer model to updated Cummins 4xCO2 relationships, including many more CMIP6 models as provided by Hege-Beate Fredriksen
- inclusion of top of atmosphere energy imbalance and ocean heat content change as diagnostic outputs
- following from above, an energy balance representation of the three-layer model, that can also be run in standalone mode
- inclusion of stochastic temperatures and forcing, introduced from Cummins et al. (2020)
- Adding back some of the emissions-driven relationships from FaIR v1.6 (land use change, ozone)
- changing the interface to a object oriented design
- an AR6-consistent constrained calibration
- everything very, very parallel

## installation

### requirements
- `anaconda` for `python3`
- `python` 3.6+
- for the Cummins calibration, a reasonably modern version of `R` (4.1.1 used here)

### python and jupyter notebooks
```
conda env create -f environment.yml
conda activate fair2.1-calibrate
nbstripout --install

# the last step is then to install this local version of fair itself
pip install -e .
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

## examples

Once `fair` is installed, the notebooks in the `examples` directory give some very incomplete tutorials on how to run the model.

## CMIP6 and AR6 calibrations

The other notebooks directory is `cmip6-ar6-calibrations` which first calibrates the model to CMIP6 and then provides a constrained probablistic distribution that is in line with the IPCC's Sixth Assessment Report, Working Group 1. There are also some model testing notebooks here (mostly related to efficiency and parallelisation).

To reproduce:

1. The `010_concatenate-hege-data-4xCO2.ipynb` notebook is run first.
2. Then, `r_scripts/calibrate_cummins_3layer.r` is run.
3. Then, `r_scripts/calibrate_cummins_2layer.r` is run.
3. Finally, the remaining notebooks are run in numerical order.

### the R scripts

Open the R console and set the working directory to the `r_scripts` directory of your local repository.

```
source("setup.r")
source("calibrate_cummins_3layer.r")
```

## acknowledgements and contributions

Should I ever write a calibration paper, these people should be on it.

1. Nick Leach and Stuart Jenkins for the original FaIR v2.0, which hopefully isn't too mangled or complicated by this attempt.
2. Hege-Beate Fredriksen for crunching the CMIP6 4xCO2 data from many more models
3. Donald Cummins for the three-layer model tuning algorithm.
4. Bill Collins for assistance with chemistry emulations and correspondence to AerChemMIP results.
5. Zeb Nicholls for advice on constraining to assessed ranges (e.g. AR6)
