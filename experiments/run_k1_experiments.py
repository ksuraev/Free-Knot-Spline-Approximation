import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT))

import compute_psi_samples

import nurnberger
import nurnberger_mod
import one_knot_approx
import plotting
import test_functions

experiments = [
    (function, k, m)
    for function in test_functions.TEST_FUNCTIONS
    for k in [1]
    for m in range(1, 4)
]

results = []

with tqdm(experiments, desc="Experiments", unit="case") as progress:
    for function, k, m in progress:
        progress.set_postfix(function=function, k=k, m=m)

        f, f_label = test_functions.TEST_FUNCTIONS[function]
        a, b = test_functions.INTERVALS[function]

        # Compute initial spline approximation based on equidistant knots
        equidistant_theta = (a + b) / 2
        equidistant_result = one_knot_approx.psi(f, a, b, equidistant_theta, m, k + 1)
        equidistant_approx = equidistant_result["approximation"]
        equidistant_max_deviation = abs(equidistant_approx.maxdeviation()[2])
        equidistant_d_max = equidistant_result["d_max"]
        equidistant_basis = equidistant_approx.basis
        equidistant_alternance_points = equidistant_approx.alternancesequence()[0]
        equidistant_status = equidistant_result["optimal"]
        equidistant_case = equidistant_result["case"]

        # Plot the equidistant spline approximation
        plotting.plot_report(
            equidistant_approx,
            points=equidistant_basis,
            f_label=test_functions.FUNCTION_LABELS[function],
            approximation_label=rf"$s^{{\mathrm{{eq}}}}_{{{m}}}(t)$",
            title="Equidistant-knot approximation",
            file_name=f"{function}_k{k}_m{m}_equidistant",
        )

        # Compute initial discontinuous spline approximation and find starting theta
        initial_approx, initial_theta = nurnberger_mod.discontinuous_spline(
            f, a, b, k, m
        )
        if initial_theta is None:
            initial_theta = initial_approx.g.knots[1]

        initial_result = one_knot_approx.psi(f, a, b, initial_theta, m, k + 1)
        initial_approx = initial_result["approximation"]
        initial_max_deviation = abs(initial_approx.maxdeviation()[2])
        initial_d_max = initial_result["d_max"]
        initial_basis = initial_approx.basis
        initial_alternance_points = initial_result[
            "approximation"
        ].alternancesequence()[0]
        initial_status = initial_result["optimal"]
        initial_case = initial_result["case"]

        # plot initial spline approximation
        plotting.plot_report(
            initial_approx,
            points=initial_basis,
            f_label=test_functions.FUNCTION_LABELS[function],
            approximation_label=rf"$s^{0}_{{{m}}}(t)$",
            title=r"Initial approximation at $\theta^{\mathrm{mod}}$",
            file_name=f"{function}_k{k}_m{m}_initial",
        )

        # Use precomputed psi(theta) values from the .npz file to get thetas and psi_values
        npz_path = Path(f"psi_surfaces/psi_surface_{function}_k{k}_m{m}.npz")
        if npz_path.exists():
            data = np.load(npz_path)
            thetas = data["theta_values"]
            psi_values = data["psi_values"]

        else:
            thetas, psi_values = compute_psi_samples.compute_psi_samples_1d(
                f, a, b, m, k + 1, a + 0.1, b - 0.1, step=0.01
            )
            np.savez(npz_path, theta_values=thetas, psi_values=psi_values)

        # plot psi(theta)
        theta_sample_min, psi_sample_min = plotting.plot_objective_psi(
            thetas, psi_values, file_name=f"psi_{function}_k{k}_m{m}"
        )

        # Find optimal theta
        theta_opt, result, iterations, theta_path = one_knot_approx.find_optimal_theta(
            f, a, b, m, k + 1, initial_theta
        )
        psi_opt = result["d_max"]

        nurnberger_original = nurnberger.run(f, a, b, k, m).g.knots[1]
        nurnberger_modified = initial_theta

        plotting.plot_objective_psi(
            thetas,
            psi_values,
            nurnbergers_orig_point=nurnberger_original,
            nurnbergers_mod_point=nurnberger_modified,
            title="Nürnberger initial points",
            file_name=f"psi_{function}_k{k}_m{m}_nurnberger_points",
        )

        # Plot psi(theta) again, this time highlighting the optimal theta found by the algorithm
        # plotting.plot_objective_psi(
        #     thetas,
        #     psi_values,
        #     theta_found=theta_opt,
        #     psi_found=psi_opt,
        #     file_name=f"psi_opt_{function}_k{k}_m{m}",
        # )

        # Plot psi(theta) again, this time highlighting the optimal theta found by the algorithm and the path taken by the algorithm
        plotting.plot_objective_psi(
            thetas,
            psi_values,
            theta_found=theta_opt,
            psi_found=psi_opt,
            theta_path=theta_path,
            title=r"Descent path on $\overline{\Psi}(\theta)$",
            file_name=f"psi_opt_path_{function}_k{k}_m{m}",
        )

        final_approx = result["approximation"]
        final_basis = final_approx.basis
        final_max_deviation = abs(final_approx.maxdeviation()[2])
        final_alternance_points = final_approx.alternancesequence()
        final_basis_points = final_approx.basis
        final_d_max = result["d_max"]
        final_gra_d_max = result.get("gra_d_max", None)
        final_status = result["optimal"]
        final_case = result["case"]

        # Plot the final spline approximation
        plotting.plot_report(
            result["approximation"],
            points=final_basis,
            f_label=test_functions.FUNCTION_LABELS[function],
            approximation_label=rf"$s^*_{{{m}}}(t)$",
            title="Final approximation",
            file_name=f"{function}_k{k}_m{m}_final",
        )

        result = {
            "function": function,
            "m": m,
            "k": k,
            "equidistantmaxdeviation": equidistant_max_deviation,
            "equidistantdmax": equidistant_d_max,
            "equidistantstatus": equidistant_status,
            "equidistantcase": equidistant_case,
            "equidistanttheta": equidistant_theta,
            "initialstatus": initial_status,
            "initialcase": initial_case,
            "initialtheta": initial_theta,
            "initialmaxdeviation": initial_max_deviation,
            "initialdmax": initial_d_max,
            "thetastart": initial_theta,
            "thetahat": theta_opt,
            "psithetahat": psi_opt,
            "thetasamplemin": theta_sample_min,
            "psithetastar": psi_sample_min,
            "finalstatus": final_status,
            "finalcase": result["case"],
            "finaldmax": final_d_max,
            "finalgradmax": final_gra_d_max,
            "finalmaxdeviation": final_max_deviation,
            "iterations": iterations,
        }
        results.append(result)


df = pd.DataFrame(results)

function_order = list(test_functions.TEST_FUNCTIONS.keys())

function_labels = {
    name: rf"$f_{{{i}}}$" for i, name in enumerate(function_order, start=1)
}

df["function_label"] = df["function"].map(function_labels)

df["function"] = pd.Categorical(
    df["function"],
    categories=function_order,
    ordered=True,
)

df = df.sort_values(["function", "m", "k"]).reset_index(drop=True)

rows = []

for i, row in df.iterrows():
    rows.append(
        f"{row['function_label']} & "
        f"{int(row['m'])} & "
        f"{int(row['k'])} & "
        f"{row['equidistantdmax']:.4f} & "
        f"{row['initialdmax']:.4f} & "
        f"{row['finaldmax']:.4f} & "
        f"{int(row['iterations'])} \\\\"
    )

    next_function = df.loc[i + 1, "function"] if i + 1 < len(df) else None

    if next_function != row["function"] and i + 1 < len(df):
        rows.append(r"\addlinespace[0.5em]")

with open("experiments/k1results_rows.tex", "w") as f:
    f.write("\n".join(rows))
