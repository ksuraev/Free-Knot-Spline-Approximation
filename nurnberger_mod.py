import numpy as np

import nurnberger
import plotting
import remez
import Spline
import test_functions

ALTERNANCE_TOL = 1e-4
TOL = 1e-4


def subroutine(f, x_i, b, m, d_n, max_iter):
    """Find x_min as the last alternance point in the interval [x_i, x_bar] such that d(x_i, x_bar) = d_n."""
    d_i, _ = nurnberger.d(f, x_i, b, m)

    # If the deviation on [x_i, b] is less than or equal to d_n, then no new knot can be placed in this interval
    if d_i <= d_n:
        return None

    # Set upper and lower bounds for bisection search
    x_l = x_i
    x_u = b

    approx = None

    for _ in range(max_iter):
        x_bar = (x_l + x_u) / 2

        # Compute the maximum deviation on [x_i, x_bar] via Remez algorithm
        d_i_max, approx_i = nurnberger.d(f, x_i, x_bar, m)

        # If the deviation is close enough to the target, we have found x_min
        if abs(d_i_max - d_n) <= TOL:
            approx = approx_i
            break

        # Determine which half of the interval to keep based on the deviation
        if d_i_max <= d_n + TOL:
            x_l = x_bar
        else:
            x_u = x_bar

        approx = approx_i

    # x_min is the (m-2)nd alternance point in the interval [x_i, x_bar]
    alt_pts, _ = approx.alternancesequence()
    if len(alt_pts) < m + 2:
        x_min = approx.basis[-1]
    else:
        x_min = alt_pts[m + 1]

    return x_min


def discontinuous_spline(f, a, b, k, m, max_iter=100, verbose=False):
    """Construct the modified Nurnberger free-knot spline approximation."""
    knots, d_min, d_max = nurnberger.step_zero(f, a, b, k, m)

    x_min = None
    d_n = None

    approximations = None

    for iteration in range(max_iter):
        if abs(d_max - d_min) < TOL * d_max:
            break

        # target deviation for this iteration computed as geometric mean
        d_n = (d_min * d_max) ** 0.5

        new_knots = [a]
        new_approximations = []
        x_i = a

        # Subroutine: solve d(x_i, x_bar) = d_n while knots can be placed
        for i in range(k):
            x_min = subroutine(f, x_i, b, m, d_n, max_iter)
            if x_min is None:
                break

            new_knots.append(x_min)

            # spline interval is [x_i, x_min].
            approx_i = remez.remez(f, x_i, x_min, m, verbose=verbose)
            new_approximations.append(approx_i)

            x_i = x_min

        # Final real interval [x_i, b].
        c_n, final_approx = nurnberger.d(f, x_i, b, m)

        new_knots.append(b)
        new_approximations.append(final_approx)

        knots = np.array(new_knots)
        approximations = new_approximations

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    # Handle convergence before the first iteration
    if approximations is None:
        approximations = [
            remez.remez(f, knots[i], knots[i + 1], m) for i in range(len(knots) - 1)
        ]

    # Create the final spline and approximation
    S = Spline.Spline(knots, [approx.g for approx in approximations])
    basis = [approx.basis for approx in approximations]
    approx = Spline.Approximation(f, S, (a, b), basis=basis)

    if verbose:
        print(f"Final max abs deviation: {abs(approx.maxdeviation()[2]):.5f}")
        print(f"Final knots: {approx.g.knots}")
        print(f"Final basis: {approx.basis}")
        print(f"x_min: {x_min}")

    return approx, x_min


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = 0, 12
    k = 1  # number of free knots (not including a and b)
    m = 1  # degree of polynomial to fit

    approx, x_min = discontinuous_spline(f, a, b, k, m)

    # plotting.plot_duo(
    #     approx,
    #     points=approx.basis,
    #     f_label=f_label,
    #     approximation_label="Piecewise polynomial approximation",
    #     points_label="Alternance points",
    #     title=f"Degree-{m} approximation with {len(approx.g.knots) - 2} free knot(s).",
    #     file_name=f"nberger_mod_{function_name}_a{a}_b{b}_k{k}_m{m}.png",
    # )
