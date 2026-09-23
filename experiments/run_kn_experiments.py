import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

results_path = Path("experiments/knresults.csv")

if results_path.exists():
    results_path.unlink()

sys.path.insert(0, str(ROOT))

import extras
import nurnberger_mod
import plotting
import test_functions

experiments = [
    (function, k, m)
    for function in test_functions.TEST_FUNCTIONS
    for k in range(3, 5)
    for m in range(1, 4)
]
results = []

with tqdm(experiments, desc="Experiments", unit="case") as progress:
    for function, k, m in progress:
        progress.set_postfix(function=function, k=k, m=m)

        f, f_label = test_functions.TEST_FUNCTIONS[function]
        a, b = test_functions.INTERVALS[function]

        # Compute initial spline approximation based on equidistant knots
        equidistant_knots = np.linspace(a, b, k + 2)
        equidistant_z, equidistant_approx, _ = extras.solve_simplex(
            f, equidistant_knots, m
        )
        # Plot the initial spline approximation
        plotting.plot_report(
            equidistant_approx,
            points=equidistant_approx.basis,
            f_label=test_functions.FUNCTION_LABELS[function],
            approximation_label=rf"$s^{{\mathrm{{eq}}}}_{{{m}}}(t)$",
            title="Equidistant-knot approximation",
            file_name=f"{function}_k{k}_m{m}_equidistant",
        )
        # Compute initial approximation using Nurnberger's modified algorithm as starting point for the descent algorithm
        modified_approx, _ = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
        initial_knots = modified_approx.g.knots
        if len(initial_knots) != k + 2:
            initial_knots = extras.insert_extra_knots(initial_knots, k, a, b)
        initial_z, initial_approx, _ = extras.solve_simplex(f, initial_knots, m)
        plotting.plot_report(
            initial_approx,
            points=initial_approx.basis,
            f_label=test_functions.FUNCTION_LABELS[function],
            approximation_label=rf"$s^{0}_{{{m}}}(t)$",
            title=r"Initial approximation at $\theta^{\mathrm{mod}}$",
            file_name=f"{function}_k{k}_m{m}_initial",
        )

        # Find optimal knots
        opt_knots, S, final_approx, iterations, final_z, iterates = extras.descent_algo(
            initial_knots[1:-1], f, a, b, m, k, track_iterates=True
        )

        # Plot the final spline approximation
        plotting.plot_report(
            final_approx,
            points=final_approx.basis,
            f_label=test_functions.FUNCTION_LABELS[function],
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
df.to_csv("experiments/knresults.csv", index=False)

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

with open("experiments/knresults_rows.tex", "w") as f:
    f.write("\n".join(rows))
