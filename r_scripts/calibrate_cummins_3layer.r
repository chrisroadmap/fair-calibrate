# Goes through each of the models in turn and tunes the parameters of the
# Cummins three layer model.

# References:
# Cummins, Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
# Estimation of Stochastic Energy Balance Model Parameters, Journal of Climate,
# 33(18), 7909-7926, https://doi.org/10.1175/JCLI-D-19-0589.1
#
# donaldcummins. (2021). donaldcummins/EBM: Optional quadratic penalty
# (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.5217975

input_data=read.csv('../data/netcdf-scm/4xCO2.csv')

inits3 <- list(
    gamma = 2.0,
    C = c(5.0, 20.0, 100.0),
    kappa = c(1.0, 2.0, 1.0),
    epsilon = 1.0,
    sigma_eta = 0.5,
    sigma_xi = 0.5,
    F_4xCO2 = 5.0
)

# prepare empty output dataframe
output <- data.frame(Doubles=double(),
                 Ints=integer(),
                 Factors=factor(),
                 Logicals=logical(),
                 Characters=character(),
                 stringsAsFactors=FALSE)

for (model in unique(input_data$climate_model)){
	rndt = input_data[(input_data$climate_model == model) & (input_data$variable == 'rndt'), 10:159]
	rndt <- unname(rndt)
	
	tas = input_data[(input_data$climate_model == model) & (input_data$variable == 'tas'), 10:159]
	tas <- unname(tas)
	
	# only for single-run models
	tas <- unlist(tas)
	rndt <- unlist(rndt)

	result <- FitKalman(inits3, T1=tas, N=rndt)$p
	check <- capture.output(result <- FitKalman(inits3, T1=tas, N=rndt)$p)
	
	if (strsplit(check, " ")[[1]][1] == "Success!") {
		conv <- TRUE
		nit <- strsplit(check, " ")[[1]][5]
	} else {
		conv <- FALSE
		nit <- NA
	}
	
	row_out <- c(model, conv, nit, result$gamma, result$C[1], result$C[2], result$C[3],
	result$kappa, result$epsilon, result$sigma_eta, result$sigma$xi, result$F_4xCO2)
}