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
from matplotlib.lines import Line2D
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

valid_temp_af = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_af_pass.csv"
).astype(np.int64)

input_ensemble_size = len(valid_temp_af)

assert input_ensemble_size > output_ensemble_size

temp_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "temperature_1850-2025.npy"
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
    "concentration_co2_2024.npy"
)
ecs_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy"
)
tcr_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy"
)
faer_in = fari_in + faci_in


def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3, 5)).x


# Indicators 2024, in prep.
gsat_params = scipy.optimize.root(opt, [1, 1, 1], args=(0.93, 1.09, 1.20)).x

samples = {}
samples["ECS"] = scipy.stats.skewnorm.rvs(
    ecs_params[0],
    loc=ecs_params[1],
    scale=ecs_params[2],
    size=10**5,
    random_state=91603,
)
samples["TCR"] = scipy.stats.norm.rvs(
    loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
# note fair produces, and we here report, total earth energy uptake, not just ocean
# this value from IGCC 2024. Use new uncertainties for ocean, assume same uncertainties
# for land, atmosphere and cryopshere.
# looking at new 2024 data from Matt Palmer, it seems unchanged from 1971-2020.
samples["OHC"] = scipy.stats.norm.rvs(
    loc=465.3, scale=108.5 / NINETY_TO_ONESIGMA, size=10**5, random_state=43178
)
samples["temperature 2005-2024"] = scipy.stats.skewnorm.rvs(
    gsat_params[0],
    loc=gsat_params[1],
    scale=gsat_params[2],
    size=10**5,
    random_state=19387,
)
# the below commented out bit is if we were to do temperature assessment using HadCRUT5 rather than Blair's assessment
#samples["temperature 2005-2024"] = scipy.stats.norm.rvs(
#    loc = 1.103988,
#    scale = 0.076402,
#    size=10**5,
#    random_state=19387,
#)
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

# IGCC 2024 forthcoming
# as this is not in yet I am doubling the uncertainty range
samples["CO2 concentration"] = scipy.stats.norm.rvs(
    loc=422.77, scale=0.4, size=10**5, random_state=81693
)

ar_distributions = {}
for constraint in [
    "ECS",
    "TCR",
    "OHC",
    "temperature 2005-2024",
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

accepted = pd.DataFrame(
    {
        "ECS": ecs_in[valid_temp_af],
        "TCR": tcr_in[valid_temp_af],
        "OHC": ohc_in[valid_temp_af] / 1e21,
        "temperature 2005-2024": np.average(
            temp_in[155:176, valid_temp_af], weights=weights_20yr, axis=0
        )
        - np.average(temp_in[:52, valid_temp_af], weights=weights_51yr, axis=0),
        "ERFari": fari_in[valid_temp_af],
        "ERFaci": faci_in[valid_temp_af],
        "ERFaer": faer_in[valid_temp_af],
        "CO2 concentration": co2_in[valid_temp_af],
    },
    index=valid_temp_af,
)

print(accepted)

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
post1_ecs = scipy.stats.gaussian_kde(ecs_in[valid_temp_af])
post2_ecs = scipy.stats.gaussian_kde(draws[0]["ECS"])

target_tcr = scipy.stats.gaussian_kde(samples["TCR"])
prior_tcr = scipy.stats.gaussian_kde(tcr_in)
post1_tcr = scipy.stats.gaussian_kde(tcr_in[valid_temp_af])
post2_tcr = scipy.stats.gaussian_kde(draws[0]["TCR"])

target_temp = scipy.stats.gaussian_kde(samples["temperature 2005-2024"])
prior_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[155:176, :], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, :], weights=weights_51yr, axis=0)
)
post1_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[155:176, valid_temp_af], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, valid_temp_af], weights=weights_51yr, axis=0)
)
post2_temp = scipy.stats.gaussian_kde(draws[0]["temperature 2005-2024"])

target_ohc = scipy.stats.gaussian_kde(samples["OHC"])
prior_ohc = scipy.stats.gaussian_kde(ohc_in / 1e21)
post1_ohc = scipy.stats.gaussian_kde(ohc_in[valid_temp_af] / 1e21)
post2_ohc = scipy.stats.gaussian_kde(draws[0]["OHC"])

target_aer = scipy.stats.gaussian_kde(samples["ERFaer"])
prior_aer = scipy.stats.gaussian_kde(faer_in)
post1_aer = scipy.stats.gaussian_kde(faer_in[valid_temp_af])
post2_aer = scipy.stats.gaussian_kde(draws[0]["ERFaer"])

target_aci = scipy.stats.gaussian_kde(samples["ERFaci"])
prior_aci = scipy.stats.gaussian_kde(faci_in)
post1_aci = scipy.stats.gaussian_kde(faci_in[valid_temp_af])
post2_aci = scipy.stats.gaussian_kde(draws[0]["ERFaci"])

target_ari = scipy.stats.gaussian_kde(samples["ERFari"])
prior_ari = scipy.stats.gaussian_kde(fari_in)
post1_ari = scipy.stats.gaussian_kde(fari_in[valid_temp_af])
post2_ari = scipy.stats.gaussian_kde(draws[0]["ERFari"])

target_co2 = scipy.stats.gaussian_kde(samples["CO2 concentration"])
prior_co2 = scipy.stats.gaussian_kde(co2_in)
post1_co2 = scipy.stats.gaussian_kde(co2_in[valid_temp_af])
post2_co2 = scipy.stats.gaussian_kde(draws[0]["CO2 concentration"])

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

if plots:
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    fig, ax = pl.subplots(3, 3, figsize=(18 / 2.54, 18 / 2.54))
    start = 0
    stop = 8
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ecs(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ecs(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ecs(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        target_ecs(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 0].set_xlim(start, stop)
    ax[0, 0].set_ylim(0, 0.6)
    ax[0, 0].set_title("ECS")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("°C")
    ax[0, 0].set_ylabel("Probability density")

    start = 0
    stop = 4
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        prior_tcr(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post1_tcr(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post2_tcr(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        target_tcr(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 1].set_xlim(start, stop)
    ax[0, 1].set_ylim(0, 1.5)
    ax[0, 1].set_title("TCR")
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xlabel("°C")

    start = 0.65
    stop = 1.45
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        target_temp(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        prior_temp(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post1_temp(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post2_temp(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 2].set_xlim(start, stop)
    ax[0, 2].set_ylim(0, 6)
    ax[0, 2].set_title("Temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 2005-2024 minus 1850-1900")

    start = -1.0
    stop = 0.4
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        target_ari(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ari(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ari(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ari(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 0].set_xlim(start, stop)
    ax[1, 0].set_ylim(0, 2.5)
    ax[1, 0].set_title("Aerosol ERFari")
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    ax[1, 0].set_ylabel("Probability density")

    start = -2.25
    stop = 0.25
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        target_aci(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        prior_aci(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post1_aci(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post2_aci(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 1].set_xlim(start, stop)
    ax[1, 1].set_ylim(0, 1.1)
    ax[1, 1].set_title("Aerosol ERFaci")
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = -3
    stop = 0.4
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        target_aer(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        prior_aer(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post1_aer(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post2_aer(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 2].set_xlim(start, stop)
    ax[1, 2].set_ylim(0, 1.1)
    ax[1, 2].set_title("Aerosol ERF")
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = 419
    stop = 427
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        target_co2(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        prior_co2(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post1_co2(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post2_co2(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[2, 0].set_xlim(start, stop)
    ax[2, 0].set_ylim(0, 1.0)
    ax[2, 0].set_ylabel("Probability density")
    ax[2, 0].set_title("CO$_2$ concentration")
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xlabel("ppm, 2024")

    start = 100
    stop = 900
    ax[2, 1].plot(
        np.linspace(start, stop),
        target_ohc(np.linspace(start, stop)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        prior_ohc(np.linspace(start, stop)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post1_ohc(np.linspace(start, stop)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post2_ohc(np.linspace(start, stop)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[2, 1].set_xlim(start, stop)
    ax[2, 1].set_ylim(0, 0.007)
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2020 minus 1971")

    ax[2, 2].axis("off")
    legend_lines = [
        Line2D([0], [0], color=colors["prior"], lw=2),
        Line2D([0], [0], color=colors["post1"], lw=2),
        Line2D([0], [0], color=colors["post2"], lw=2),
        Line2D([0], [0], color=colors["target"], lw=2),
    ]
    legend_labels = ["Prior", "Temperature RMSE", "All constraints", "Target"]
    ax[2, 2].legend(legend_lines, legend_labels, frameon=False, loc="upper left")

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
    pl.scatter(draws[0]["TCR"], draws[0]["ECS"])
    pl.xlabel("TCR, °C")
    pl.ylabel("ECS, °C")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ecs_tcr_constrained.png"
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
print("ECS:", np.percentile(draws[0]["ECS"], (5, 50, 95)))
print("TCR:", np.percentile(draws[0]["TCR"], (5, 50, 95)))
print(
    "CO2 concentration 2024:", np.percentile(draws[0]["CO2 concentration"], (5, 50, 95))
)
print(
    "Temperature 2005-2024 rel. 1850-1900:",
    np.percentile(draws[0]["temperature 2005-2024"], (5, 50, 95)),
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
    df_gmst = pd.read_csv("../../../../../data/forcing/IGCC_GMST_1850-2024.csv")
    gmst = df_gmst["gmst"].values

    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2026),
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
        np.arange(1850, 2026),
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
        np.arange(1850, 2026),
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
        np.arange(1850, 2026),
        np.median(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2025), gmst, color="b", label="Observations")

    ax.legend(frameon=False, loc="upper left")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Constrained, reweighted posterior")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_reweighted_historical.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_reweighted_historical.pdf"
    )
    pl.close()

np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    sorted(draws[0].index),
    fmt="%d",
)
