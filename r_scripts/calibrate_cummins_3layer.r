# Goes through each of the models in turn and tunes the parameters of the
# Cummins three layer model.
#
# It will produce an output csv table of Cummins three layer parameters, which
# we will then put in impulse-response form for FaIR.
#
# References:
# Cummins, Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
# Estimation of Stochastic Energy Balance Model Parameters, Journal of Climate,
# 33(18), 7909-7926, https://doi.org/10.1175/JCLI-D-19-0589.1
#
# donaldcummins. (2021). donaldcummins/EBM: Optional quadratic penalty
# (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.5217975

input_data = read.csv("../data/netcdf-scm/4xCO2.csv")

inits3 <- list(gamma = 2, C = c(5, 20, 100), kappa = c(1, 2, 1), epsilon = 1, sigma_eta = 0.5, 
	sigma_xi = 0.5, F_4xCO2 = 5)

# prepare empty output dataframe
output <- data.frame(matrix(ncol = 15, nrow = 0))

models = unique(input_data$climate_model)

for (model in models) {
	runs <- input_data[(input_data$climate_model == model) & (input_data$variable == 
		"tas"), 3]

	for (run in runs) {
		rndt <- input_data[(input_data$climate_model == model) & (input_data$variable == 
			"rndt") & (input_data$member_id == run), 10:159]
		rndt <- unname(rndt)

		tas <- input_data[(input_data$climate_model == model) & (input_data$variable == 
			"tas") & (input_data$member_id == run), 10:159]
		tas <- unname(tas)

		tas <- unlist(tas)
		rndt <- unlist(rndt)

        # PUT A TRYCATCH HERE TO DEAL WITH CONVERGENCE AND ITERATION ERRORS
		check <- capture.output(result <- FitKalman(inits3, T1 = tas, N = rndt, alpha = 1e-05)$p)
		print(check)

		if (strsplit(check, " ")[[1]][1] == "Success!") {
			conv <- TRUE
			nit <- strsplit(check, " ")[[1]][5]
		} else {
			conv <- FALSE
			nit <- NA
		}
		row_out <- c(model, run, conv, nit, result$gamma, result$C[1], result$C[2], 
			result$C[3], result$kappa[1], result$kappa[2], result$kappa[3], result$epsilon, 
			result$sigma_eta, result$sigma_xi, result$F_4xCO2)

		output <- rbind(output, row_out)
	}
}

names = c("model", "run", "conv", "nit", "gamma", "C1", "C2", "C3", "kappa1", "kappa2", 
	"kappa3", "epsilon", "sigma_eta", "sigma_xi", "F_4xCO2")
colnames(output) <- names

# save output
ifelse(!dir.exists(file.path("..", "data", "calibration")), dir.create(file.path("..", 
	"data", "calibration")), FALSE)

write.csv(output, "../data/calibration/4xCO2_cummins.csv")
