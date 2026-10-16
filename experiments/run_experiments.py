import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

results_path = SCRIPT_DIR / "results.csv"

if results_path.exists():
    results_path.unlink()

sys.path.insert(0, str(ROOT))


import multi_knot_approx
import nurnberger_mod
import plotting
import test_functions

KNOT_TYPES = ("x_min", "x_max", "x_min_rev", "x_max_rev")

experiments = [
    (function, k, m)
    for function in test_functions.TEST_FUNCTIONS
    for k in range(1, 5)
    for m in range(1, 4)
]


def _plot_initial_knots_approx(
    f, function, function_label, k, m, knots, nsamples, knot_type=""
):
    if len(knots) != k:
        knots = [a, *knots, b]
        print(
            f"Number of knots ({len(knots)}) for type '{knot_type}' does not match expected number ({k + 2}). Inserting extra knots."
        )
        knots = multi_knot_approx.insert_extra_knots(knots, k)
    else:
        knots = [a, *knots, b]

    # Compute the initial approximation and its maximum deviation
    _, _, approx, z = multi_knot_approx.evaluate_and_get_direction(
        f, knots, m, nsamples=nsamples
    )

    # Plot the initial spline approximation
    # plotting.plot_single(
    #     approx,
    #     points=approx.basis,
    #     f_label=function_label,
    #     approximation_label=rf"$s^{0}_{{{m}}}(t)$",
    #     title=rf"Initial approximation at $\theta^{{\mathrm{{mod}}}}$ ({knot_type})",
    #     file_name=f"{function}_k{k}_m{m}_initial_{knot_type}",
    # )
    return approx, z


def _plot_final_knots_approx(
    f,
    function,
    function_label,
    a,
    b,
    k,
    m,
    knots,
    nsamples,
    knot_type="",
):
    if len(knots) != k:
        knots = [a, *knots, b]
        print(
            f"Number of knots ({len(knots)}) for type '{knot_type}' does not match expected number ({k + 2}). Inserting extra knots."
        )
        knots = multi_knot_approx.insert_extra_knots(knots, k)
        knots = knots[1:-1]

    opt_knots, S, approx, iterations, final_z = multi_knot_approx.descent_algorithm(
        knots, f, a, b, m, k, nsamples=nsamples, track_iterates=False
    )

    # Plot the final spline approximation
    plotting.plot_single(
        approx,
        points=approx.basis,
        f_label=function_label,
        approximation_label=rf"$s^*_{{{m}}}(t)$",
        title=rf"Final approximation ({knot_type})",
        file_name=f"{function}_k{k}_m{m}_final_{knot_type}",
    )
    return opt_knots, S, approx, iterations, final_z


def analyse_alternance(nsamples, approx, S, opt_knots):
    alt_pts = approx.basis
    counts = []
    n_intervals = len(opt_knots) - 1

    for i in range(n_intervals):
        left = opt_knots[i]
        right = opt_knots[i + 1]

        if i == n_intervals - 1:
            # Last interval -  [left, right]
            count = np.sum((alt_pts >= left) & (alt_pts <= right))
        else:
            # Other intervals - [left, right)
            count = np.sum((alt_pts >= left) & (alt_pts < right))

        counts.append(int(count))

    knot_is_alt_pt = []
    differentiable = []

    for j, knot in enumerate(opt_knots[1:-1], start=1):
        samples = np.linspace(a, b, nsamples)
        closest_sample = samples[np.argmin(np.abs(samples - knot))]
        knot_is_alt_pt.append(
            bool(np.any(np.isclose(alt_pts, closest_sample, atol=1e-12, rtol=0)))
        )

        differentiable.append(S.is_differentiable_at(j))

    return {
        "alt_pts": alt_pts,
        "alt_per_interval": counts,
        "knot_is_alt_pt": knot_is_alt_pt,
        "differentiable": differentiable,
    }


with tqdm(experiments, desc="Experiments", unit="case") as progress:
    for function, k, m in progress:
        progress.set_postfix(function=function, k=k, m=m)

        f, _ = test_functions.TEST_FUNCTIONS[function]
        function_label = test_functions.FUNCTION_LABELS[function]
        a, b = test_functions.INTERVALS[function]

        # Compute initial approximation at Nurnberger's modified points
        x_min_knots, x_max_knots = nurnberger_mod.get_knots(f, a, b, k, m)

        x_min_knots_rev, x_max_knots_rev = nurnberger_mod.get_knots_reverse(
            f, a, b, k, m
        )

        knot_sets = {
            "x_min": x_min_knots,
            "x_max": x_max_knots,
            "x_min_rev": x_min_knots_rev,
            "x_max_rev": x_max_knots_rev,
        }

        initial_results = {}
        nsamples = 1000
        for knot_type, knots in knot_sets.items():
            _, z = _plot_initial_knots_approx(
                f,
                function,
                function_label,
                k,
                m,
                knots,
                nsamples,
                knot_type=knot_type,
            )

            initial_results[knot_type] = z

        final_results = {}

        for knot_type, knots in knot_sets.items():
            opt_knots, S, approx, iterations, z = _plot_final_knots_approx(
                f,
                function,
                function_label,
                a,
                b,
                k,
                m,
                knots,
                nsamples,
                knot_type=knot_type,
            )

            # Get alternance points for the final approximation
            alternance = analyse_alternance(nsamples, approx, S, opt_knots)

            final_results[knot_type] = {
                "knots": opt_knots,
                "iterations": iterations,
                "z": z,
                **alternance,
            }

        result = {"function": function, "m": m, "k": k}
        for knot_type in KNOT_TYPES:
            result[f"{knot_type}_knots"] = [
                round(float(x), 2) for x in final_results[knot_type]["knots"][1:-1]
            ]

        for knot_type in KNOT_TYPES:
            result[f"{knot_type}_initial_z"] = round(
                float(abs(initial_results[knot_type])),
                3,
            )

        for knot_type in KNOT_TYPES:
            result[f"{knot_type}_final_z"] = round(
                float(abs(final_results[knot_type]["z"])),
                3,
            )

        for knot_type in KNOT_TYPES:
            result[f"{knot_type}_alt_per_interval"] = final_results[knot_type][
                "alt_per_interval"
            ]

            result[f"{knot_type}_knot_is_alt_pt"] = final_results[knot_type][
                "knot_is_alt_pt"
            ]

            result[f"{knot_type}_differentiable"] = final_results[knot_type][
                "differentiable"
            ]

            result[f"{knot_type}_alt_pts"] = [
                round(float(x), 2) for x in final_results[knot_type]["alt_pts"]
            ]

        for knot_type in KNOT_TYPES:
            result[f"{knot_type}_iterations"] = final_results[knot_type]["iterations"]

        pd.DataFrame([result]).to_csv(
            results_path, mode="a", header=not results_path.exists(), index=False
        )
