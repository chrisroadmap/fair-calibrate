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
from fair.earth_params import mass_atmosphere, molecular_weight_air
from tqdm.auto import tqdm

pl.switch_backend("agg")

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
    "ocean_heat_content_2020_minus_1971.npy"
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
    "concentration_co2_2022.npy"
)
# ecs_in = np.load(
#     f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy"
# )
tcr_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy"
)
eei_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "toa_imbalance_2013-2022_mean.npy"
)
af_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "airborne_fraction_1pctCO2_y70_y140.npy"
)
faer_in = fari_in + faci_in


def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


# ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3, 5)).x
gsat_params = scipy.optimize.root(opt, [1, 1, 1], args=(0.87, 1.03, 1.13)).x

samples = {}
# samples["ECS"] = scipy.stats.skewnorm.rvs(
#     ecs_params[0],
#     loc=ecs_params[1],
#     scale=ecs_params[2],
#     size=10**5,
#     random_state=91603,
# )
samples["TCR"] = scipy.stats.norm.rvs(
    loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
samples["EEI"] = scipy.stats.norm.rvs(
    loc=1.7, scale=0.8 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
# note fair produces, and we here report, total earth energy uptake, not just ocean
# this value from IGCC 2023. Use new uncertainties for ocean, assume same uncertainties
# for land, atmosphere and cryopshere.
samples["OHC"] = scipy.stats.norm.rvs(
    loc=465.3, scale=108.5 / NINETY_TO_ONESIGMA, size=10**5, random_state=43178
)
samples["temperature 2003-2022"] = scipy.stats.skewnorm.rvs(
    gsat_params[0],
    loc=gsat_params[1],
    scale=gsat_params[2],
    size=10**5,
    random_state=19387,
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
# IGCC paper: 417.1 +/- 0.4
# IGCC dataset: 416.9
# my assessment 417.0 +/- 0.5
samples["CO2 concentration"] = scipy.stats.norm.rvs(
    loc=417.0, scale=0.5, size=10**5, random_state=81693
)

ar_distributions = {}
for constraint in [
#    "ECS",
    "TCR",
    "EEI",
    "OHC",
    "temperature 2003-2022",
    "ERFari",
    "ERFaci",
    "ERFaer",
    "CO2 concentration",
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
co2_1920 = co2_1850 * 1.01**70  # NOT 2x (69.66 yr), per definition of TCRE
mass_factor = 12.011 / molecular_weight_air * mass_atmosphere / 1e21

accepted = pd.DataFrame(
    {
#        "ECS": ecs_in[valid_temp],
        "TCR": tcr_in[valid_temp],
        "EEI": eei_in[valid_temp],
        "OHC": ohc_in[valid_temp] / 1e21,
        "temperature 2003-2022": np.average(
            temp_in[153:174, valid_temp], weights=weights_20yr, axis=0
        )
        - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0),
        "ERFari": fari_in[valid_temp],
        "ERFaci": faci_in[valid_temp],
        "ERFaer": faer_in[valid_temp],
        "CO2 concentration": co2_in[valid_temp],
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

# target_ecs = scipy.stats.gaussian_kde(samples["ECS"])
# prior_ecs = scipy.stats.gaussian_kde(ecs_in)
# post1_ecs = scipy.stats.gaussian_kde(ecs_in[valid_temp])
# post2_ecs = scipy.stats.gaussian_kde(draws[0]["ECS"])

target_tcr = scipy.stats.gaussian_kde(samples["TCR"])
prior_tcr = scipy.stats.gaussian_kde(tcr_in)
post1_tcr = scipy.stats.gaussian_kde(tcr_in[valid_temp])
post2_tcr = scipy.stats.gaussian_kde(draws[0]["TCR"])

target_eei = scipy.stats.gaussian_kde(samples["EEI"])
prior_eei = scipy.stats.gaussian_kde(eei_in)
post1_eei = scipy.stats.gaussian_kde(eei_in[valid_temp])
post2_eei = scipy.stats.gaussian_kde(draws[0]["EEI"])

target_temp = scipy.stats.gaussian_kde(samples["temperature 2003-2022"])
prior_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[153:174, :], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, :], weights=weights_51yr, axis=0)
)
post1_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[153:174, valid_temp], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0)
)
post2_temp = scipy.stats.gaussian_kde(draws[0]["temperature 2003-2022"])

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

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

if plots:
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    # fig, ax = pl.subplots(3, 3, figsize=(10, 10))
    # start = 0
    # stop = 8
    # ax[0, 0].plot(
    #     np.linspace(start, stop, 1000),
    #     prior_ecs(np.linspace(start, stop, 1000)),
    #     color=colors["prior"],
    #     label="Prior",
    # )
    # ax[0, 0].plot(
    #     np.linspace(start, stop, 1000),
    #     post1_ecs(np.linspace(start, stop, 1000)),
    #     color=colors["post1"],
    #     label="Temperature RMSE",
    # )
    # ax[0, 0].plot(
    #     np.linspace(start, stop, 1000),
    #     post2_ecs(np.linspace(start, stop, 1000)),
    #     color=colors["post2"],
    #     label="All constraints",
    # )
    # ax[0, 0].plot(
    #     np.linspace(start, stop, 1000),
    #     target_ecs(np.linspace(start, stop, 1000)),
    #     color=colors["target"],
    #     label="Target",
    # )
    # ax[0, 0].set_xlim(start, stop)
    # ax[0, 0].set_ylim(0, 0.5)
    # ax[0, 0].set_title("ECS")
    # ax[0, 0].set_yticklabels([])
    # ax[0, 0].set_xlabel("°C")

    fig, ax = pl.subplots(3, 3, figsize=(10, 10))
    start = 0.5
    stop = 3
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        prior_eei(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post1_eei(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post2_eei(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        target_eei(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
    )
    ax[0, 0].set_xlim(start, stop)
    ax[0, 0].set_ylim(0, 2)
    ax[0, 0].set_title("TOA imbalance")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("W m$^{-2}$, 2013-2022")

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

    start = 0.6
    stop = 1.4
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
    ax[0, 2].set_ylim(0, 6)
    ax[0, 2].set_title("Temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 2003-2022 minus 1850-1900")

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

    start = 413
    stop = 421
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
    ax[2, 0].set_xlabel("ppm, 2022")

    start = 100
    stop = 900
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
    ax[2, 1].set_ylim(0, 0.007)
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2020 minus 1971")

    ax[2, 2].axis("off")

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "constraints.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "constraints.pdf"
    )
    pl.close()


if plots:
    pl.scatter(draws[0]["TCR"], draws[0]["ERFaci"] + draws[0]["ERFari"])
    pl.xlabel("TCR, °C")
    pl.ylabel("Aerosol ERF, W m$^{-2}$, 2005-2014 minus 1750")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "tcr_aer_constrained.png"
    )
    pl.close()


# move these to the validation script
print("Constrained, reweighted parameters:")
#print("ECS:", np.percentile(draws[0]["ECS"], (5, 50, 95)))
print("TCR:", np.percentile(draws[0]["TCR"], (5, 50, 95)))
print("TOA imbalance 2013-2022:", np.percentile(draws[0]["EEI"], (5, 50, 95)))
print(
    "CO2 concentration 2022:", np.percentile(draws[0]["CO2 concentration"], (5, 50, 95))
)
print(
    "Temperature 2003-2022 rel. 1850-1900:",
    np.percentile(draws[0]["temperature 2003-2022"], (5, 50, 95)),
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
print("OHC change 2020 rel. 1971:", np.percentile(draws[0]["OHC"], (5, 50, 95)))

print("*likely range")

if plots:
    df_gmst = pd.read_csv("../../../../../data/forcing/IGCC_GMST_1850-2022.csv")
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

    ax.plot(np.arange(1850.5, 2023), gmst, color="b", label="Observations")

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
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_reweighted_ssp245.pdf"
    )
    pl.close()

np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    sorted(draws[0].index),
    fmt="%d",
)
