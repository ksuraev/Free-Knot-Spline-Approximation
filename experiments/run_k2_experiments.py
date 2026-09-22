import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT))

import compute_psi_samples

import extras
import nurnberger
import nurnberger_mod
import plotting
import test_functions

experiments = [
    (function, k, m)
    for function in test_functions.TEST_FUNCTIONS
    for k in [2]
    for m in range(4, 7)
]

results = []

with tqdm(experiments, desc="Experiments", unit="case") as progress:
    for function, k, m in progress:
        progress.set_postfix(function=function, k=k, m=m)

        f, f_label = test_functions.TEST_FUNCTIONS[function]
        a, b = test_functions.INTERVALS[function]

        # Compute initial spline approximation based on equidistant knots
        initial_knots = np.linspace(a, b, k + 2)
        initial_z, initial_approx, initial_signs = extras.solve_simplex(
            f, initial_knots, m
        )
        # Plot the initial spline approximation
        plotting.plot_report(
            initial_approx,
            points=initial_approx.basis,
            f_label=f_label,
            approximation_label=rf"$S_{{{m}}}(t)$",
            file_name=f"{function}_k{k}_m{m}_initial",
        )

        modified_approx, _ = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
        nurnberger_modified = modified_approx.g.knots[1:-1]

        if k == 2:

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

            # plot psi(theta) as 3d surface
            plotting.plot_objective_psi_3d(
                theta_1_values,
                theta_2_values,
                psi_values,
                file_name=f"psi_{function}_k{k}_m{m}",
            )

            # plot psi(theta) as contour plot
            plotting.plot_objective_psi_contour(
                theta_1_values,
                theta_2_values,
                psi_values,
                file_name=f"psi_contour_{function}_k{k}_m{m}",
            )

            # Plot psi(theta) as contour plot with Nurnberger's original and modified points
            nurnberger_original = nurnberger.run(f, a, b, k, m).g.knots[1:-1]

            plotting.plot_objective_psi_contour(
                theta_1_values,
                theta_2_values,
                psi_values,
                nurnbergers_orig_point=nurnberger_original,
                nurnbergers_mod_point=nurnberger_modified,
                file_name=f"psi_contour_{function}_k{k}_m{m}_nurnberger_points",
            )

        # Find optimal knots
        opt_knots, S, final_approx, iterations, iterates = extras.descent_algo(
            nurnberger_modified, f, a, b, m, k, track_iterates=True
        )

        if k == 2:

            # Plot psi(theta) again, this time highlighting the optimal theta found by the algorithm and the path taken by the algorithm
            plotting.plot_objective_psi_3d(
                theta_1_values,
                theta_2_values,
                psi_values,
                theta_found=opt_knots[1:-1],
                theta_path=iterates,
                file_name=f"psi_path_{function}_k{k}_m{m}",
            )

            plotting.plot_objective_psi_contour(
                theta_1_values,
                theta_2_values,
                psi_values,
                theta_found=opt_knots[1:-1],
                theta_path=iterates,
                file_name=f"psi_path_contour_{function}_k{k}_m{m}",
            )

        # Plot the final spline approximation
        plotting.plot_report(
            final_approx,
            points=final_approx.basis,
            f_label=f_label,
            approximation_label=rf"$S_{{{m}}}(t)$",
            file_name=f"{function}_k{k}_m{m}_final",
        )

        result = {
            "function": function,
            "m": m,
            "k": k,
            "initialmaxdeviation": initial_z,
            "theta_initial": initial_knots[1:-1],
            "thetastart": nurnberger_modified,
            "thetaopt": opt_knots[1:-1],
            "finalmaxdeviation": final_approx.maxdeviation()[2],
            "iterations": iterations,
        }
        results.append(result)

df = pd.DataFrame(results)

df.to_csv("experiments/k2results.csv", index=False)
