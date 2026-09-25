import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

results_path = Path("experiments/k2results.csv")

if results_path.exists():
    results_path.unlink()

sys.path.insert(0, str(ROOT))

import compute_psi_samples

import multi_knot_approx
import nurnberger
import nurnberger_mod
import plotting
import test_functions

experiments = [
    (function, k, m)
    for function in test_functions.TEST_FUNCTIONS
    for k in range(2, 3)
    for m in range(1, 4)
]
results = []

with tqdm(experiments, desc="Experiments", unit="case") as progress:
    for function, k, m in progress:
        progress.set_postfix(function=function, k=k, m=m)

        f, _ = test_functions.TEST_FUNCTIONS[function]
        function_label = test_functions.FUNCTION_LABELS[function]
        a, b = test_functions.INTERVALS[function]

        # Compute initial spline approximation based on equidistant knots
        equidistant_knots = np.linspace(a, b, k + 2)
        _, _, equidistant_approx, equidistant_z = (
            multi_knot_approx.evaluate_and_get_direction(f, equidistant_knots, m)
        )

        # Plot the equidistant spline approximation
        plotting.plot_single(
            equidistant_approx,
            points=equidistant_approx.basis,
            f_label=function_label,
            approximation_label=rf"$s^{{\mathrm{{eq}}}}_{{{m}}}(t)$",
            title="Equidistant-knot approximation",
            file_name=f"{function}_k{k}_m{m}_equidistant",
        )

        # Compute initial approximation at Nurnberger's modified points
        modified_approx, _ = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
        initial_knots = modified_approx.g.knots

        # If the number of internal knots is not equal to k, insert extra knots
        if len(initial_knots) != k + 2:
            initial_knots = multi_knot_approx.insert_extra_knots(initial_knots, k)

        # Compute the initial approximation and its maximum deviation
        _, _, initial_approx, initial_z = multi_knot_approx.evaluate_and_get_direction(
            f, initial_knots, m
        )

        # Plot the initial spline approximation
        plotting.plot_single(
            initial_approx,
            points=initial_approx.basis,
            f_label=function_label,
            approximation_label=rf"$s^{0}_{{{m}}}(t)$",
            title=r"Initial approximation at $\theta^{\mathrm{mod}}$",
            file_name=f"{function}_k{k}_m{m}_initial",
        )

        # Use precomputed psi(theta) values from the .npz file to get thetas and psi_values
        npz_path = Path(f"psi_surfaces/psi_surface_{function}_k{k}_m{m}.npz")
        if npz_path.exists():
            data = np.load(npz_path)
            theta_1_values = data["theta_1_values"]
            theta_2_values = data["theta_2_values"]
            psi_values = data["psi_values"]

        else:
            theta_1_values, theta_2_values, psi_values = (
                compute_psi_samples.compute_psi_samples_2d(
                    f, a, b, m, a + 0.1, b - 0.1, a + 0.1, b - 0.1
                )
            )
            np.savez(
                npz_path,
                theta_1_values=theta_1_values,
                theta_2_values=theta_2_values,
                psi_values=psi_values,
            )

        # # plot Psi bar(theta) as 3d surface
        # plotting.plot_objective_psi_bar_3d(
        #     theta_1_values,
        #     theta_2_values,
        #     psi_values,
        #     title=r"Objective surface $\overline{\Psi}(\theta)$",
        #     file_name=f"psi_{function}_k{k}_m{m}",
        # )

        # # plot Psi bar(theta) as contour plot
        # plotting.plot_objective_psi_bar_contour(
        #     theta_1_values,
        #     theta_2_values,
        #     psi_values,
        #     title=r"Objective contours $\overline{\Psi}(\theta)$",
        #     file_name=f"psi_contour_{function}_k{k}_m{m}",
        # )

        # Plot Psi bar(theta) as contour plot with Nurnberger's original and modified points
        nurnberger_original = nurnberger.run(f, a, b, k, m).g.knots[1:-1]

        plotting.plot_objective_psi_bar_contour(
            theta_1_values,
            theta_2_values,
            psi_values,
            nurnbergers_orig_point=nurnberger_original,
            nurnbergers_mod_point=initial_knots[1:-1],
            title="Nürnberger initial points",
            file_name=f"psi_contour_{function}_k{k}_m{m}_nurnberger_points",
        )

        # Find optimal knots
        opt_knots, S, final_approx, iterations, final_z, iterates = (
            multi_knot_approx.descent_algorithm(
                initial_knots[1:-1], f, a, b, m, k, track_iterates=True
            )
        )

        # Plot Psi bar(theta) again, this time highlighting the optimal theta found by the algorithm and the path taken by the algorithm
        plotting.plot_objective_psi_bar_3d(
            theta_1_values,
            theta_2_values,
            psi_values,
            theta_found=opt_knots[1:-1],
            theta_path=iterates,
            title=r"Descent path on $\overline{\Psi}(\theta)$",
            file_name=f"psi_path_{function}_k{k}_m{m}",
        )

        plotting.plot_objective_psi_bar_contour(
            theta_1_values,
            theta_2_values,
            psi_values,
            theta_found=opt_knots[1:-1],
            theta_path=iterates,
            title=r"Descent path on $\overline{\Psi}(\theta)$",
            file_name=f"psi_path_contour_{function}_k{k}_m{m}",
        )

        # Plot the final spline approximation
        plotting.plot_single(
            final_approx,
            points=final_approx.basis,
            f_label=function_label,
            approximation_label=rf"$s^*_{{{m}}}(t)$",
            title="Final approximation",
            file_name=f"{function}_k{k}_m{m}_final",
        )

        result = {
            "function": function,
            "m": m,
            "k": k,
            "thetaequidistant": equidistant_knots[1:-1].tolist(),
            "thetastart": initial_knots[1:-1].tolist(),
            "thetaopt": opt_knots[1:-1].tolist(),
            "equidistantmaxdeviation": abs(equidistant_z),
            "initialmaxdeviation": abs(initial_z),
            "finalmaxdeviation": abs(final_z),
            "iterations": iterations,
        }
        results.append(result)
        pd.DataFrame([result]).to_csv(
            results_path,
            mode="a",
            header=not results_path.exists(),
            index=False,
        )

df = pd.DataFrame(results)
df.to_csv("experiments/k2results.csv", index=False)

function_order = list(test_functions.TEST_FUNCTIONS.keys())

df["function_label"] = df["function"].map(test_functions.FUNCTION_LABELS)

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
        f"{row['equidistantmaxdeviation']:.4f} & "
        f"{row['initialmaxdeviation']:.4f} & "
        f"{row['finalmaxdeviation']:.4f} & "
        f"{int(row['iterations'])} \\\\"
    )

    next_function = df.loc[i + 1, "function"] if i + 1 < len(df) else None

    if next_function != row["function"] and i + 1 < len(df):
        rows.append(r"\addlinespace[0.5em]")

with open("experiments/k2results_rows.tex", "w") as f:
    f.write("\n".join(rows))
