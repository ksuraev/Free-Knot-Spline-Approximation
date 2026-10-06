import numpy as np

import nurnberger
import plotting
import remez
import Spline
import test_functions

ALTERNANCE_TOL = 1e-4
TOL = 1e-4


def subroutine_xmin(f, x_i, b, m, d_n, max_iter):
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

    # sometimes alt sequence returns fewer than m+2 points
    if len(alt_pts) < m + 2:
        x_min = approx.basis[-1]
    else:
        x_min = alt_pts[m + 1]

    return x_min


def subroutine_xmin_xmax(f, x_i, b, m, d_n, max_iter):
    d_i, _ = nurnberger.d(f, x_i, b, m)

    # If the deviation on [x_i, b] is less than or equal to d_n, then no new knot can be placed in this interval
    if d_i <= d_n:
        return None, None

    # Set upper and lower bounds for bisection search
    x_l = x_i
    x_u = b

    for _ in range(max_iter):
        x_bar = (x_l + x_u) / 2

        # Compute the maximum deviation on [x_i, x_bar]
        d_i_max, approx = nurnberger.d(f, x_i, x_bar, m)

        # If the upper and lower bounds are sufficiently close, we have found x_max
        if x_u - x_l < TOL:
            break

        # Determine which half of the interval to keep based on the deviation
        if d_i_max <= d_n + TOL:
            x_l = x_bar
        else:
            x_u = x_bar

    # x_max is the upper bound of the last interval where the deviation was less than or equal to d_n
    x_max = x_l

    # Approximation on [x_i, x_max] is used to find the (m + 2)-th alternance point
    _, approx = nurnberger.d(f, x_i, x_max, m)
    alt_pts, _ = approx.alternancesequence()

    if len(alt_pts) < m + 2:
        x_min = approx.basis[m + 1]
    else:
        x_min = alt_pts[m + 1]  # (m + 2)-th alternance point

    return x_min, x_max


def get_knots(f, a, b, k, m, max_iter=100, verbose=False):
    """Compute the x_min and x_max knots starting from the left."""
    knots, d_min, d_max = nurnberger.step_zero(f, a, b, k, m)

    x_min_knots = []
    x_max_knots = []

    for iteration in range(max_iter):
        if abs(d_max - d_min) < TOL * d_max:
            break

        # target deviation for this iteration computed as geometric mean
        d_n = (d_min * d_max) ** 0.5

        current_x_min_knots = []
        current_x_max_knots = []

        x_i = a

        # Subroutine: solve d(x_i, x_bar) = d_n while knots can be placed
        for i in range(k):
            x_min, x_max = subroutine_xmin_xmax(f, x_i, b, m, d_n, max_iter)
            if x_min is None or x_max is None:
                break

            current_x_min_knots.append(x_min)
            current_x_max_knots.append(x_max)

            x_i = x_min

        # Final real interval [x_i, b].
        c_n, _ = nurnberger.d(f, x_i, b, m)

        x_min_knots = current_x_min_knots
        x_max_knots = current_x_max_knots

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    if verbose:
        print(f"x_min knots: {x_min_knots}")
        print(f"x_max knots: {x_max_knots}")

    return x_min_knots, x_max_knots


def subroutine_xmin_xmax_reverse(f, a, x_i, m, d_n, max_iter):
    d_i, _ = nurnberger.d(f, a, x_i, m)

    if d_i <= d_n:
        return None, None

    # Set upper and lower bounds for bisection search
    x_l = a
    x_u = x_i

    for _ in range(max_iter):
        x_bar = (x_l + x_u) / 2

        # Compute maximum deviation on [x_bar, x_i]
        d_i_max, _ = nurnberger.d(f, x_bar, x_i, m)

        if x_u - x_l < TOL:
            break

        if d_i_max <= d_n + TOL:
            x_u = x_bar
        else:
            x_l = x_bar

    # x_min is the left endpoint where the deviation is d_n
    x_min = x_u

    # Recompute the approximation on [x_min, x_i]
    _, approx = nurnberger.d(f, x_min, x_i, m)

    # x_max is the (m + 2)-th alternance point from the right
    alt_pts, _ = approx.alternancesequence()
    if len(alt_pts) < m + 2:
        x_max = approx.basis[-(m + 2)]
    else:
        x_max = alt_pts[-(m + 2)]

    return x_min, x_max


def get_knots_reverse(f, a, b, k, m, max_iter=100, verbose=False):
    """Compute the x_min and x_max knots starting from the right."""
    knots, d_min, d_max = nurnberger.step_zero(f, a, b, k, m)

    x_min_knots = []
    x_max_knots = []

    for iteration in range(max_iter):
        if abs(d_max - d_min) < TOL * d_max:
            break

        # target deviation for this iteration computed as geometric mean
        d_n = (d_min * d_max) ** 0.5

        current_x_min_knots = []
        current_x_max_knots = []

        x_i = b

        # Subroutine: solve d(x_bar, x_i) = d_n while knots can be placed
        for _ in range(k):
            x_min, x_max = subroutine_xmin_xmax_reverse(f, a, x_i, m, d_n, max_iter)
            if x_min is None or x_max is None:
                break

            current_x_min_knots.append(x_min)
            current_x_max_knots.append(x_max)

            x_i = x_max

        # Final real interval [a, x_i].
        c_n, _ = nurnberger.d(f, a, x_i, m)

        # Reverse the order of knots
        current_x_min_knots.reverse()
        current_x_max_knots.reverse()

        x_min_knots = current_x_min_knots
        x_max_knots = current_x_max_knots

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    if verbose:
        print(f"x_min knots (reverse): {x_min_knots}")
        print(f"x_max knots (reverse): {x_max_knots}")

    return x_min_knots, x_max_knots


def discontinuous_spline(f, a, b, k, m, max_iter=100, verbose=False):
    """Construct the modified Nurnberger free-knot spline approximation"""
    knots, d_min, d_max = nurnberger.step_zero(f, a, b, k, m)

    x_min_knots = []
    x_max_knots = []

    approximations = None

    for iteration in range(max_iter):
        if abs(d_max - d_min) < TOL * d_max:
            break

        # target deviation for this iteration computed as geometric mean
        d_n = (d_min * d_max) ** 0.5

        new_knots = [a]
        new_approximations = []

        current_x_min_knots = []
        current_x_max_knots = []

        x_i = a

        # Subroutine: solve d(x_i, x_bar) = d_n while knots can be placed
        for i in range(k):
            x_min, x_max = subroutine_xmin_xmax(f, x_i, b, m, d_n, max_iter)
            if x_min is None or x_max is None:
                break

            current_x_min_knots.append(x_min)
            current_x_max_knots.append(x_max)

            # x_min is the actual spline knot
            new_knots.append(x_min)

            # Calculate the spline approximation on [x_i, x_min]
            approx_i = remez.run(f, x_i, x_min, m)
            new_approximations.append(approx_i)

            x_i = x_min

        # Final real interval [x_i, b].
        c_n, final_approx = nurnberger.d(f, x_i, b, m)

        new_knots.append(b)
        new_approximations.append(final_approx)

        knots = new_knots
        approximations = new_approximations

        x_min_knots = current_x_min_knots
        x_max_knots = current_x_max_knots

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    # Handle convergence before the first iteration
    if approximations is None:
        approximations = [
            remez.run(f, knots[i], knots[i + 1], m) for i in range(len(knots) - 1)
        ]

    # Create the final spline and approximation
    S = Spline.Spline(knots, [approx.g for approx in approximations])
    basis = [approx.basis for approx in approximations]
    approx = Spline.Approximation(f, S, (a, b), basis=basis)

    if verbose:
        print(f"Final max abs deviation: {abs(approx.maxdeviation()[2]):.5f}")
        print(f"Final basis: {approx.basis}")
        print(f"x_min knots: {x_min_knots}")
        print(f"x_max knots: {x_max_knots}")

    return approx, x_min_knots, x_max_knots


if __name__ == "__main__":
    function_name = "cos_if_else"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = test_functions.INTERVALS[function_name]
    k = 4  # number of free knots (not including a and b)
    m = 2  # degree of polynomial to fit

    approx, x_min_knots, x_max_knots = discontinuous_spline(f, a, b, k, m, verbose=True)

    x_min_knots, x_max_knots = get_knots(f, a, b, k, m, max_iter=100, verbose=True)

    x_min_knots_rev, x_max_knots_rev = get_knots_reverse(
        f, a, b, k, m, max_iter=100, verbose=True
    )
