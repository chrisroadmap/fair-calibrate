# Alternative Cummins calibration strategies

These scripts are **not part of `./run`** and are not maintained. They are kept
because they implement calibration strategies other than the one in
`input/calibration/02_calibrate_cummins_3layer.r`.

| Script | Strategy |
| --- | --- |
| `calibrate_cummins_2layer.r` | Cummins two-layer energy balance model fitted to the CMIP6 abrupt-4xCO2 tables |
| `calibrate_cummins_3layer_longrunmip.r` | Cummins three-layer model fitted to LongRunMIP abrupt-4xCO2 runs |

Last known to work at tag `v1.6.1`, from `r_scripts/`. They predate the current
flat output layout: they read the `CALIBRATION_VERSION`, `FAIR_VERSION` and
(2-layer only) `CONSTRAINT_SET` environment variables, which the pipeline no
longer uses, and they expect `output/fair-<version>/v<calibration>/...` paths.
Before running, port the paths to `output/calibrations/` (relative to their new
location) and remove the environment variable reads. Run them inside the pixi
environment (`pixi run`).
