#!/usr/bin/env Rscript

# Goes through each of the models in turn and tunes the parameters of the
# Cummins two layer model.
#
# It will produce an output csv table of Cummins two layer parameters, which
# we will then put in impulse-response form for FaIR.
#
# References:
# Cummins, Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
# Estimation of Stochastic Energy Balance Model Parameters, Journal of Climate,
# 33(18), 7909-7926, https://doi.org/10.1175/JCLI-D-19-0589.1
#
# donaldcummins. (2021). donaldcummins/EBM: Optional quadratic penalty
# (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.5217975

# Use Donald Cummins' package
library(EBM)

# Get environment variable describing calibration version
readRenviron("../.env")
cal_v = paste("v", Sys.getenv("CALIBRATION_VERSION"), sep="")
fair_v = paste("fair-", Sys.getenv("FAIR_VERSION"), sep="")

# Get the precalculated 4xCO2 N and T data
input_data = read.csv(file.path("..", "output", fair_v, cal_v, "calibrations", "4xCO2_cmip6.csv"))

# Initial guess for parameter values
inits2 <- list(
	gamma = 2,
	C = c(7.5, 75),
	kappa = c(1, 0.8),
	epsilon = 1.2,
	sigma_eta = 0.5,
	sigma_xi = 0.5,
	F_4xCO2 = 8
)

# prepare empty output dataframe
output <- data.frame(matrix(ncol = 13, nrow = 0))

# grab models
models = unique(input_data$climate_model)

# iterate through models
for (model in models) {
	runs <- input_data[
		(input_data$climate_model == model) &
		(input_data$variable == "tas"), 3
	]

	# iterate through runs of the same model for now, though we will probably
	# want to combine/downweight this in the final edition
	for (run in runs) {
		rndt <- input_data[
			(input_data$climate_model == model) &
			(input_data$variable == "rndt") &
			(input_data$member_id == run), 10:159
		]
		rndt <- unname(rndt)

		tas <- input_data[
			(input_data$climate_model == model) &
			(input_data$variable ==  "tas") &
			(input_data$member_id == run), 10:159
		]
		tas <- unname(tas)

		tas <- unlist(tas)
		rndt <- unlist(rndt)

        # Try and achieve convergence: if it fails, use a different alpha and try again.
		# If it still fails, give up and move on.
		attempt <- 0
		success <- FALSE
		while (attempt < 5) {
			tryCatch(
				{
					check <- capture.output(
						result <- FitKalman(
							inits2,
							T1 = tas,
							N = rndt,
							alpha = 1e-05 * 10^attempt,
							maxeval=20000)$p
						)
					success <- TRUE
					print(check)
				},
				error = function(c) {
					message(paste(model, "did not converge or ran out of iterations"))
					message("Here's the original error message from FitKalman:")
					message(c)
					if (attempt < 4) {
						message("Trying again with more liberal quadratic penalty (alpha)")
					}
					else {
						message(paste("This isn't going to work, is it? I'm giving up for", model))
					}
				}
			)
			if (success) { break }
			attempt <- attempt + 1
		}
		if (!success) {
			row_out <- c(model, run, FALSE, NA, NA, NA, NA, NA, NA,
			NA, NA, NA, NA)

			output <- rbind(output, row_out)
			next
		}

		conv <- TRUE
		nit <- strsplit(check, " ")[[1]][5]

		row_out <- c(
			model,
			run,
			conv,
			nit,
			result$gamma,
			result$C[1],
			result$C[2],
			result$kappa[1],
			result$kappa[2],
			result$epsilon,
			result$sigma_eta,
			result$sigma_xi,
			result$F_4xCO2
		)

		output <- rbind(output, row_out)
	}
}

# rename columns away from random defaults
names = c("model", "run", "conv", "nit", "gamma", "C1", "C2", "kappa1",
	"kappa2", "epsilon", "sigma_eta", "sigma_xi", "F_4xCO2")
colnames(output) <- names

# save output
write.csv(
    output,
    file.path("..", "output", fair_v, cal_v, "calibrations","4xCO2_cummins_ebm2_cmip6.csv"),
    row.names=FALSE
)
