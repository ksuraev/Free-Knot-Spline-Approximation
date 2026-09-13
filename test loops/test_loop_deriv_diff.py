# test candidate internal knots
# bisect the interval of internal knots on the pair where the derivative difference changes sign
import csv

import numpy as np

import nadia
import test_functions

TOL = 1e-5

if __name__ == "__main__":
    function_name = "sin"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = 2, 6
    k = 1  # number of internal fixed knots
    m = 2  # degree of polynomial to fit in each subinterval
    n = k + 1  # number of subintervals

    # Choose intial knots
    candidate_knots = np.arange(3.4, 3.5, 0.01)
    results = []

    for internal_knot in candidate_knots:
        knots = [2, internal_knot, 6]

        result = nadia.gra(f, knots, m, n)

        coeffs = result["coefficients"]

        left_deriv = coeffs[0, 0] + 2 * coeffs[0, 1] * (knots[1] - knots[0])
        right_deriv = coeffs[1, 0]

        deriv_difference = right_deriv - left_deriv
        differentiable = abs(deriv_difference) <= TOL

        if differentiable:
            print(f"Differentiable at internal knot {internal_knot}.")

        results.append(
            {
                "function": function_name,
                "optimal": result["optimal"],
                "m": m,
                "k": k,
                "internal_knot": internal_knot,
                "max_abs_deviation": result["d_max"],
                "deriv_difference": deriv_difference,
                "differentiable": abs(deriv_difference) <= TOL,
                # "a0": np.round(result["a0"], 6),
                # "coefficients": np.round(result["coefficients"], 6).tolist(),
                # "t_star": np.round(result["t_star"], 6),
                # "delta": np.round(result["delta"], 8),
                # "basis": np.round(np.concatenate(result["basis"]), 6).tolist(),
                # "alternance_points": np.round(result["alternance_points"], 6).tolist(),
            }
        )

    # Write results to CSV
    with open(f"{function_name}_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # find the pair where the derivative difference sign changes, use that as the initial interval for bisection, get midpoint, run algo, calculate derivative, check if differentiable, if not - keep the half where the sign change occurs

    # find pair of internal knots where sign of derivative difference changes
    left = None
    right = None
    left_diff = None
    right_diff = None

    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]

        d1 = r1["deriv_difference"]
        d2 = r2["deriv_difference"]

        if d1 * d2 < 0:
            left = r1["internal_knot"]
            left_diff = d1
            right = r2["internal_knot"]
            right_diff = d2
            break

    if left is None:
        print("No pair of internal knots where the derivative difference changes sign")
    else:
        print(f"Bisection interval: [{left}, {right}]")

        while right - left > TOL:
            mid = (left + right) / 2

            knots = [2, mid, 6]

            result = nadia.gra(f, knots, m, n)

            a = result["coefficients"]
            left_deriv = a[0, 0] + 2 * a[0, 1] * (knots[1] - knots[0])
            right_deriv = a[1, 0]
            mid_diff = right_deriv - left_deriv

            print(
                f"knot = {mid:.8f}, mid_diff = {mid_diff:.8f}, max abs deviation = {result['d_max']:.8f}"
            )

            if abs(mid_diff) <= TOL:
                print(f"Differentiable at knot {mid:.8f}")
                print(f"Max abs deviation: {result['d_max']:.8f}")
                break

            # Keep the half containing the sign change
            if left_diff * mid_diff < 0:
                right = mid
                right_diff = mid_diff
            else:
                left = mid
                left_diff = mid_diff
