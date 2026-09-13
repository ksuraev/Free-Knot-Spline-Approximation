# very dodgy way to find minimum of d_max over a range of internal knots
import csv

import numpy as np

import nadia
import nadia_mod
import test_functions

TOL = 1e-5


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 1
    n = k + 1
    m = 2

    candidate_knots = np.arange(-0.99, 0.99, 0.01)
    results = []

    for internal_knot in candidate_knots:
        knots = [a, internal_knot, b]

        result = nadia.gra(f, knots, m, n)

        results.append(
            {
                "function": function_name,
                "optimal": result["optimal"],
                "m": m,
                "k": k,
                "internal_knot": internal_knot,
                "max_abs_deviation": result["d_max"],
                "chain": result["chain"],
                # "a0": np.round(result["a0"], 6),
                # "coefficients": np.round(result["coefficients"], 6).tolist(),
                # "t_star": np.round(result["t_star"], 6),
                # "delta": np.round(result["delta"], 8),
                # "basis": np.round(np.concatenate(result["basis"]), 6).tolist(),
                # "alternance_points": np.round(result["alternance_points"], 6).tolist(),
            }
        )

    with open(f"{function_name}_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # candidate_knots = np.arange(3.67, 3.69, 0.001)

    # results = []
    # for internal_knot in candidate_knots:
    #     knots = [a, internal_knot, b]
    #     result = nadia_exchange_mod.gra(f, knots, m, n)
    #     results.append((internal_knot, result["d_max"]))

    # best_knot, best_deviation = min(results, key=lambda r: r[1])
    # print(f"Best knot: {best_knot:.8f}")
    # print(f"Max abs deviation: {best_deviation:.8f}")
