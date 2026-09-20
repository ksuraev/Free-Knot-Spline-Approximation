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
import plotting
import test_functions

experiments = [
    (function, k, m)
    for function in test_functions.TEST_FUNCTIONS
    for k in [2]
    for m in range(1, 4)
]

results = []

with tqdm(experiments, desc="Experiments", unit="case") as progress:
    for function, k, m in progress:
        progress.set_postfix(function=function, k=k, m=m)

        f, f_label = test_functions.TEST_FUNCTIONS[function]
        a, b = test_functions.INTERVALS[function]

        # Compute initial spline approximation based on equidistant knots

        # Plot the initial spline approximation

        # # Compute initial discontinuous spline approximation and find starting theta
        # approx, theta_start = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
        # if theta_start is None:
        #     theta_start = approx.g.knots[1:-1]

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
        nurnberger_modified, _ = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
        nurnberger_modified = nurnberger_modified.g.knots[1:-1]

        plotting.plot_objective_psi_contour(
            theta_1_values,
            theta_2_values,
            psi_values,
            nurnbergers_orig_point=nurnberger_original,
            nurnbergers_mod_point=nurnberger_modified,
            file_name=f"psi_contour_{function}_k{k}_m{m}_nurnberger_points",
        )

        # Find optimal knots

        # Plot psi(theta) again, this time highlighting the optimal found by the algorithm

        # Plot psi(theta) again, this time highlighting the optimal theta found by the algorithm and the path taken by the algorithm

        # Plot the final spline approximation

        # result = {
        #     "function": function,
        #     "m": m,
        #     "k": k,
        #     "initialstatus": initial_status,
        #     "initialcase": initial_case,
        #     "initialtheta": initial_theta,
        #     "initialmaxdeviation": initial_max_deviation,
        #     "thetastart": theta_start,
        #     "thetahat": theta_opt,
        #     "psithetahat": psi_opt,
        #     "thetasamplemin": theta_sample_min,
        #     "psithetastar": psi_sample_min,
        #     "finalstatus": final_status,
        #     "finalmaxdeviation": final_max_deviation,
        #     "iterations": iterations,
        # }
        # results.append(result)


# df = pd.DataFrame(results)

# df.to_csv("k2results.csv", index=False)
