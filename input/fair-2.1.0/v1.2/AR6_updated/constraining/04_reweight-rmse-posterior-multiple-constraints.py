#!/usr/bin/env python
# coding: utf-8

"""Apply posterior weighting"""

# mention in paper: skew-normal distribution
# this is where Zeb earns his corn

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
from fair.constants import DOUBLING_TIME_1PCT
from fair.earth_params import mass_atmosphere, molecular_weight_air
from tqdm.auto import tqdm

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use("../../../../../defaults.mplstyle")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

assert fair_v == __version__

print("Doing reweighting...")


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

valid_temp = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv"
).astype(np.int64)

input_ensemble_size = len(valid_temp)

assert input_ensemble_size > output_ensemble_size

temp_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "temperature_1850-2101.npy"
)
ohc_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "ocean_heat_content_2018_minus_1971.npy"
)
fari_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "forcing_ari_2005-2014_mean.npy"
)
faci_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "forcing_aci_2005-2014_mean.npy"
)
co2_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "concentration_co2_2014.npy"
)
ecs_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy"
)
tcr_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy"
)
af_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "airborne_fraction_1pctCO2_y70_y140.npy"
)
faer_in = fari_in + faci_in
#ssp245_in = np.load(
#    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
#    "temperature_ssp245_concdriven_2081-2100_mean.npy"
#)
tcre_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "temperature_1pctCO2_1000GtC.npy"
)
tcr1pct_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "temperature_1pctCO2_y70_y140.npy"
)

## Find NaNs and drop from valid_temp
## currently only an issue for 4xCO2 AF
#print("Step 1 posterior size:", len(valid_temp))
#nanrun = np.isnan(af_in[1,:])
#goodrun = ~np.isnan(af_in[1,:])
#nanidx = np.arange(samples, dtype=int)[nanrun]
#valid_temp = np.setdiff1d(valid_temp, nanidx)
#print("Step 1 posterior size after removing NaNs:", len(valid_temp))

def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    # print(q05, q50, q95, x)
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


samples = {}
samples["ECS"] = scipy.stats.skewnorm.rvs(
    8.82185594, loc=1.95059779, scale=1.55584604, size=10**5, random_state=91603
)
samples["TCR"] = scipy.stats.norm.rvs(
    loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
samples["OHC"] = scipy.stats.norm.rvs(
    loc=396 / 0.91, scale=67 / 0.91, size=10**5, random_state=43178
)
samples["temperature 1995-2014"] = scipy.stats.skewnorm.rvs(
    -1.65506091, loc=0.92708099, scale=0.12096636, size=10**5, random_state=19387
)
samples["ERFari"] = scipy.stats.norm.rvs(
    loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173
)
samples["ERFaci"] = scipy.stats.norm.rvs(
    loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123
)
samples["ERFaer"] = scipy.stats.norm.rvs(
    loc=-1.3,
    scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_ONESIGMA,
    size=10**5,
    random_state=3916153,
)
samples["CO2 concentration"] = scipy.stats.norm.rvs(
    loc=397.5469792683919, scale=0.36, size=10**5, random_state=81693
)
samples["ssp245 2081-2100"] = scipy.stats.skewnorm.rvs(
    2.20496701, loc=1.4124379, scale=0.60080822, size=10**5, random_state=801693589
)
samples["TCRE"] = scipy.stats.norm.rvs(
    loc=1.65, scale=0.65 / NINETY_TO_ONESIGMA, size=10**5, random_state=198236970
)
#samples["AF 2xCO2"] = scipy.stats.norm.rvs(
#    loc=0.53, scale=0.06, size=10**5, random_state=198236970
#)
#samples["AF 4xCO2"] = scipy.stats.norm.rvs(
#    loc=0.6, scale=0.1, size=10**5, random_state=3456711
#)

ar_distributions = {}
for constraint in [
    "ECS",
    "TCR",
    "OHC",
    "temperature 1995-2014",
    "ERFari",
    "ERFaci",
    "ERFaer",
    "CO2 concentration",
    "ssp245 2081-2100",
#    "TCRE",
#    "AF 2xCO2",
#    "AF 4xCO2"
]:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
        samples[constraint], bins=100, density=True
    )[1]
    ar_distributions[constraint]["values"] = samples[constraint]

weights_20yr = np.ones(21)
weights_20yr[0] = 0.5
weights_20yr[-1] = 0.5
weights_51yr = np.ones(52)
weights_51yr[0] = 0.5
weights_51yr[-1] = 0.5

co2_1850 = 284.3169988
co2_1920 = co2_1850*1.01**70  # NOT 2x (69.66 yr), per definition of TCRE
mass_factor = 12.011 / molecular_weight_air * mass_atmosphere / 1e21

accepted = pd.DataFrame(
    {
        "ECS": ecs_in[valid_temp],
        "TCR": tcr_in[valid_temp],
#        "TCR": tcr1pct_in[0, valid_temp] * DOUBLING_TIME_1PCT/70,
        "OHC": ohc_in[valid_temp] / 1e21,
        "temperature 1995-2014": np.average(
            temp_in[145:166, valid_temp], weights=weights_20yr, axis=0
        )
        - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0),
        "ERFari": fari_in[valid_temp],
        "ERFaci": faci_in[valid_temp],
        "ERFaer": faer_in[valid_temp],
        "CO2 concentration": co2_in[valid_temp],
        "ssp245 2081-2100": np.average(
            temp_in[231:252, valid_temp], weights=weights_20yr, axis=0
        )
        - np.average(temp_in[145:166, valid_temp], weights=weights_20yr, axis=0),
#        "TCRE": tcre_in[valid_temp],
#        "TCRE": tcr1pct_in[0, valid_temp] * af_in[0, valid_temp] / ((co2_1920-co2_1850)*mass_factor)
#        "AF 2xCO2": af_in[0, valid_temp],
#        "AF 4xCO2": af_in[1, valid_temp],
    },
    index=valid_temp,
)


def calculate_sample_weights(distributions, samples, niterations=50):
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in tqdm(
        range(niterations), desc="Iterations", leave=False, disable=1 - progress
    ):
        gofs.append([])
        if k == (niterations - 1):
            weights_second_last_iteration = weights.copy()
            weights_to_average = []

        for j, unique_code in enumerate(unique_codes):
            unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                unique_code, distributions, samples, weights, j, k
            )
            if k == (niterations - 1):
                weights_to_average.append(unique_code_weights[our_values_bin_idx])

            weights *= unique_code_weights[our_values_bin_idx]

            gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
            gofs[-1].append(gof)

            gofs_full.append([unique_code])
            for unique_code_check in unique_codes:
                unique_code_check_weights, _ = get_unique_code_weights(
                    unique_code_check, distributions, samples, weights, 1, 1
                )
                gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                gofs_full[-1].append(gof)

    weights_stacked = np.vstack(weights_to_average).mean(axis=0)
    weights_final = weights_stacked * weights_second_last_iteration

    gofs_full.append(["Final iteration"])
    for unique_code_check in unique_codes:
        unique_code_check_weights, _ = get_unique_code_weights(
            unique_code_check, distributions, samples, weights_final, 1, 1
        )
        gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
        gofs_full[-1].append(gof)

    return (
        weights_final,
        pd.DataFrame(np.array(gofs), columns=unique_codes),
        pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
    )


def get_unique_code_weights(unique_code, distributions, samples, weights, j, k):
    bin_edges = distributions[unique_code]["bins"]
    our_values = samples[unique_code].copy()

    our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
    np.testing.assert_allclose(bin_edges, bin_edges_np)
    assessed_ranges_bin_counts, _ = np.histogram(
        distributions[unique_code]["values"], bins=bin_edges
    )

    our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

    existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
    for i in range(existing_weighted_bin_counts.shape[0]):
        existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

    if np.equal(j, 0) and np.equal(k, 0):
        np.testing.assert_equal(
            existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
        )

    unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

    # existing_weighted_bin_counts[0] refers to samples outside the
    # assessed range's lower bound. Accordingly, if `our_values` was
    # digitized into a bin idx of zero, it should get a weight of zero.
    unique_code_weights[0] = 0
    # Similarly, if `our_values` was digitized into a bin idx greater
    # than the number of bins then it was outside the assessed range
    # so get a weight of zero.
    unique_code_weights[-1] = 0

    for i in range(1, our_values_bin_counts.shape[0] + 1):
        # the histogram idx is one less because digitize gives values in the
        # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
        histogram_idx = i - 1
        if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
            unique_code_weights[i] = 0
        elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
            # other variables force this box to be zero so just fill it with
            # one
            unique_code_weights[i] = 1
        else:
            unique_code_weights[i] = (
                assessed_ranges_bin_counts[histogram_idx]
                / existing_weighted_bin_counts[histogram_idx]
            )

    return unique_code_weights, our_values_bin_idx


weights, gofs, gofs_full = calculate_sample_weights(
    ar_distributions, accepted, niterations=30
)

effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
print("Number of effective samples:", effective_samples)

assert effective_samples >= output_ensemble_size

draws = []
drawn_samples = accepted.sample(
    n=output_ensemble_size, replace=False, weights=weights, random_state=10099
)
draws.append((drawn_samples))

target_ecs = scipy.stats.gaussian_kde(samples["ECS"])
prior_ecs = scipy.stats.gaussian_kde(ecs_in)
post1_ecs = scipy.stats.gaussian_kde(ecs_in[valid_temp])
post2_ecs = scipy.stats.gaussian_kde(draws[0]["ECS"])

target_tcr = scipy.stats.gaussian_kde(samples["TCR"])
prior_tcr = scipy.stats.gaussian_kde(tcr_in)
post1_tcr = scipy.stats.gaussian_kde(tcr_in[valid_temp])
#prior_tcr = scipy.stats.gaussian_kde(tcr1pct_in * DOUBLING_TIME_1PCT/70)
#post1_tcr = scipy.stats.gaussian_kde(tcr1pct_in[0, valid_temp] * DOUBLING_TIME_1PCT/70)
post2_tcr = scipy.stats.gaussian_kde(draws[0]["TCR"])

target_temp = scipy.stats.gaussian_kde(samples["temperature 1995-2014"])
prior_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[145:166, :], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, :], weights=weights_51yr, axis=0)
)
post1_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[145:166, valid_temp], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0)
)
post2_temp = scipy.stats.gaussian_kde(draws[0]["temperature 1995-2014"])

target_ssp = scipy.stats.gaussian_kde(samples["ssp245 2081-2100"])
prior_ssp = scipy.stats.gaussian_kde(
    np.average(temp_in[231:252, :], weights=weights_20yr, axis=0)
    - np.average(temp_in[145:166, :], weights=weights_20yr, axis=0)
)
post1_ssp = scipy.stats.gaussian_kde(
    np.average(temp_in[231:252, valid_temp], weights=weights_20yr, axis=0)
    - np.average(temp_in[145:166, valid_temp], weights=weights_20yr, axis=0)
)
post2_ssp  = scipy.stats.gaussian_kde(draws[0]["ssp245 2081-2100"])

target_ohc = scipy.stats.gaussian_kde(samples["OHC"])
prior_ohc = scipy.stats.gaussian_kde(ohc_in / 1e21)
post1_ohc = scipy.stats.gaussian_kde(ohc_in[valid_temp] / 1e21)
post2_ohc = scipy.stats.gaussian_kde(draws[0]["OHC"])

target_aer = scipy.stats.gaussian_kde(samples["ERFaer"])
prior_aer = scipy.stats.gaussian_kde(faer_in)
post1_aer = scipy.stats.gaussian_kde(faer_in[valid_temp])
post2_aer = scipy.stats.gaussian_kde(draws[0]["ERFaer"])

target_aci = scipy.stats.gaussian_kde(samples["ERFaci"])
prior_aci = scipy.stats.gaussian_kde(faci_in)
post1_aci = scipy.stats.gaussian_kde(faci_in[valid_temp])
post2_aci = scipy.stats.gaussian_kde(draws[0]["ERFaci"])

target_ari = scipy.stats.gaussian_kde(samples["ERFari"])
prior_ari = scipy.stats.gaussian_kde(fari_in)
post1_ari = scipy.stats.gaussian_kde(fari_in[valid_temp])
post2_ari = scipy.stats.gaussian_kde(draws[0]["ERFari"])

target_co2 = scipy.stats.gaussian_kde(samples["CO2 concentration"])
prior_co2 = scipy.stats.gaussian_kde(co2_in)
post1_co2 = scipy.stats.gaussian_kde(co2_in[valid_temp])
post2_co2 = scipy.stats.gaussian_kde(draws[0]["CO2 concentration"])

#target_tcre = scipy.stats.gaussian_kde(samples["TCRE"])
##prior_tcre = scipy.stats.gaussian_kde(tcre_in)
##post1_tcre = scipy.stats.gaussian_kde(tcre_in[valid_temp])
#prior_tcre = tcr1pct_in[0, :] * af_in[0, :] / ((co2_1920-co2_1850)*mass_factor)
#post1_tcre = tcr1pct_in[0, valid_temp] * af_in[0, valid_temp] / ((co2_1920-co2_1850)*mass_factor)
#post2_tcre = scipy.stats.gaussian_kde(draws[0]["TCRE"])

#target_af2 = scipy.stats.gaussian_kde(samples["AF 2xCO2"])
#prior_af2 = scipy.stats.gaussian_kde(af_in[0,:])
#post1_af2 = scipy.stats.gaussian_kde(af_in[0,valid_temp])
#post2_af2 = scipy.stats.gaussian_kde(draws[0]["AF 2xCO2"])

#target_af4 = scipy.stats.gaussian_kde(samples["AF 4xCO2"])
#prior_af4 = scipy.stats.gaussian_kde(af_in[1,goodrun])
#post1_af4 = scipy.stats.gaussian_kde(af_in[1,valid_temp])
#post2_af4 = scipy.stats.gaussian_kde(draws[0]["AF 4xCO2"])

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

if plots:
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    fig, ax = pl.subplots(3, 3, figsize=(10, 10))
    start = 0
    stop = 8
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ecs(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ecs(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ecs(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        target_ecs(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[0, 0].set_xlim(start, stop)
    ax[0, 0].set_ylim(0, 0.5)
    ax[0, 0].set_title("ECS")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("°C")

    start = 0
    stop = 4
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        prior_tcr(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post1_tcr(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post2_tcr(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        target_tcr(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[0, 1].set_xlim(start, stop)
    ax[0, 1].set_ylim(0, 1.4)
    ax[0, 1].set_title("TCR")
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xlabel("°C")

    start = 0.5
    stop = 1.3
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        target_temp(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        prior_temp(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post1_temp(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post2_temp(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[0, 2].set_xlim(start, stop)
    ax[0, 2].set_ylim(0, 5)
    ax[0, 2].set_title("Temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 1995-2014 minus 1850-1900")

    start = -1.0
    stop = 0.3
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        target_ari(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ari(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ari(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ari(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[1, 0].set_xlim(start, stop)
    ax[1, 0].set_ylim(0, 3)
    ax[1, 0].set_title("Aerosol ERFari")
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = -2.25
    stop = 0.25
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        target_aci(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        prior_aci(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post1_aci(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post2_aci(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[1, 1].set_xlim(start, stop)
    ax[1, 1].set_ylim(0, 1.6)
    ax[1, 1].set_title("Aerosol ERFaci")
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = -3
    stop = 0
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        target_aer(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        prior_aer(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post1_aer(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post2_aer(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[1, 2].set_xlim(start, stop)
    ax[1, 2].set_ylim(0, 1.6)
    ax[1, 2].set_title("Aerosol ERF")
    ax[1, 2].legend(frameon=False, loc="upper left")
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = 394
    stop = 402
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        target_co2(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        prior_co2(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post1_co2(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post2_co2(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[2, 0].set_xlim(start, stop)
    ax[2, 0].set_ylim(0, 1.2)
    ax[2, 0].set_title("CO$_2$ concentration")
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xlabel("ppm, 2014")

    start = 0
    stop = 800
    ax[2, 1].plot(
        np.linspace(start, stop),
        target_ohc(np.linspace(start, stop)),
        color=colors["target"],
        label="Target",
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        prior_ohc(np.linspace(start, stop)),
        color=colors["prior"],
        label="Prior",
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post1_ohc(np.linspace(start, stop)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post2_ohc(np.linspace(start, stop)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[2, 1].set_xlim(start, stop)
    ax[2, 1].set_ylim(0, 0.006)
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2018 minus 1971")

#    ax[2, 2].axis('off')

    start = 0.8
    stop = 3.2
    ax[2, 2].plot(
        np.linspace(start, stop, 1000),
        target_ssp(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[2, 2].plot(
        np.linspace(start, stop, 1000),
        prior_ssp(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[2, 2].plot(
        np.linspace(start, stop, 1000),
        post1_ssp(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[2, 2].plot(
        np.linspace(start, stop, 1000),
        post2_ssp(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[2, 2].set_xlim(start, stop)
    ax[2, 2].set_ylim(0, 1.1)
    ax[2, 2].set_title("Temperature anomaly")
    ax[2, 2].set_yticklabels([])
    ax[2, 2].set_xlabel("°C, 2081-2100 minus 1995-2014, ssp245")

#    start = 0.0
#    stop = 3.2
#    ax[2, 2].plot(
#        np.linspace(start, stop, 1000),
#        target_tcre(np.linspace(start, stop, 1000)),
#        color=colors["target"],
#        label="Target",
#    )
#    ax[2, 2].plot(
#        np.linspace(start, stop, 1000),
#        prior_tcre(np.linspace(start, stop, 1000)),
#        color=colors["prior"],
#        label="Prior",
#    )
#    ax[2, 2].plot(
#        np.linspace(start, stop, 1000),
#        post1_tcre(np.linspace(start, stop, 1000)),
#        color=colors["post1"],
#        label="Temperature RMSE",
#    )
#    ax[2, 2].plot(
#        np.linspace(start, stop, 1000),
#        post2_tcre(np.linspace(start, stop, 1000)),
#        color=colors["post2"],
#        label="All constraints",
#    )
#    ax[2, 2].set_xlim(start, stop)
#    ax[2, 2].set_ylim(0, 1.2)
#    ax[2, 2].set_title("TCRE")
#    ax[2, 2].set_yticklabels([])
#    ax[2, 2].set_xlabel("°C (1000 GtC)$^{-1}$")
#
#    start = 0.33
#    stop = 0.73
#    ax[3, 1].plot(
#        np.linspace(start, stop, 1000),
#        target_af2(np.linspace(start, stop, 1000)),
#        color=colors["target"],
#        label="Target",
#    )
#    ax[3, 1].plot(
#        np.linspace(start, stop, 1000),
#        prior_af2(np.linspace(start, stop, 1000)),
#        color=colors["prior"],
#        label="Prior",
#    )
#    ax[3, 1].plot(
#        np.linspace(start, stop, 1000),
#        post1_af2(np.linspace(start, stop, 1000)),
#        color=colors["post1"],
#        label="Temperature RMSE",
#    )
#    ax[3, 1].plot(
#        np.linspace(start, stop, 1000),
#        post2_af2(np.linspace(start, stop, 1000)),
#        color=colors["post2"],
#        label="All constraints",
#    )
#    ax[3, 1].set_xlim(start, stop)
#    ax[3, 1].set_ylim(0, 1.1)
#    ax[3, 1].set_title("Airborne fraction 2xCO2")
#    ax[3, 1].set_yticklabels([])
#    ax[3, 1].set_xlabel("[-]")
#
#    start = 0.30
#    stop = 1.00
#    ax[3, 2].plot(
#        np.linspace(start, stop, 1000),
#        target_af4(np.linspace(start, stop, 1000)),
#        color=colors["target"],
#        label="Target",
#    )
#    ax[3, 2].plot(
#        np.linspace(start, stop, 1000),
#        prior_af4(np.linspace(start, stop, 1000)),
#        color=colors["prior"],
#        label="Prior",
#    )
#    ax[3, 2].plot(
#        np.linspace(start, stop, 1000),
#        post1_af4(np.linspace(start, stop, 1000)),
#        color=colors["post1"],
#        label="Temperature RMSE",
#    )
#    ax[3, 2].plot(
#        np.linspace(start, stop, 1000),
#        post2_af4(np.linspace(start, stop, 1000)),
#        color=colors["post2"],
#        label="All constraints",
#    )
#    ax[3, 2].set_xlim(start, stop)
#    ax[3, 2].set_ylim(0, 1.1)
#    ax[3, 2].set_title("Airborne fraction 4xCO2")
#    ax[3, 2].set_yticklabels([])
#    ax[3, 2].set_xlabel("[-]")
    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "constraints.png"
    )
    pl.close()


if plots:
    pl.scatter(draws[0]["TCR"], draws[0]["ECS"])
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ecs_tcr_constrained.png"
    )
    pl.close()


if plots:
    pl.scatter(draws[0]["TCR"], draws[0]["ERFaci"] + draws[0]["ERFari"])
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "tcr_aer_constrained.png"
    )
    pl.close()


# move these to the validation script
print("Constrained, reweighted parameters:")
print("ECS:", np.percentile(draws[0]["ECS"], (5, 50, 95)))
print("TCR:", np.percentile(draws[0]["TCR"], (5, 50, 95)))
print(
    "CO2 concentration 2014:", np.percentile(draws[0]["CO2 concentration"], (5, 50, 95))
)
print(
    "Temperature 1995-2014 rel. 1850-1900:",
    np.percentile(draws[0]["temperature 1995-2014"], (5, 50, 95)),
)
print(
    "Aerosol ERFari 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFari"], (5, 50, 95)),
)
print(
    "Aerosol ERFaci 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFaci"], (5, 50, 95)),
)
print(
    "Aerosol ERF 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFaci"] + draws[0]["ERFari"], (5, 50, 95)),
)
print(
    "OHC change 2018 rel. 1971*:", np.percentile(draws[0]["OHC"] * 0.91, (16, 50, 84))
)
print("ssp245 2081-2100:", np.percentile(draws[0]["ssp245 2081-2100"], (5, 50, 95)))
#print(
#    "TCRE from 2xCO2:", np.percentile(draws[0]["TCRE"], (5, 50, 95))
#)
#print(
#    "Airborne fraction 2xCO2*:", np.percentile(draws[0]["AF 2xCO2"], (16, 50, 84))
#)
#print(
#    "Airborne fraction 4xCO2*:", np.percentile(draws[0]["AF 4xCO2"], (16, 50, 84))
#)

print("*likely range")

if plots:
    df_gmst = pd.read_csv("../../../../../data/forcing/AR6_GMST.csv")
    gmst = df_gmst["gmst"].values

    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        np.max(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            5,
            axis=1,
        ),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            95,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            16,
            axis=1,
        ),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            84,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850, 2102),
        np.median(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2021), gmst, color="b", label="Observations")

    ax.legend(frameon=False, loc="upper left")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Temperature anomaly: posterior ssp245")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_reweighted_ssp245.png"
    )
    pl.close()

np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    sorted(draws[0].index),
    fmt="%d",
)
