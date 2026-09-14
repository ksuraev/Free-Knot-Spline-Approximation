import numpy as np

import nurnberger
import plotting
import remez
import Spline
import test_functions

ALTERNANCE_TOL = 1e-4
TOL = 1e-5


def deviation(f, P, t):
    return f(t) - P(t)


def subroutine(f, x_i, b, degree, d_n, max_iter):
    """Find x_min and x_max for the next interval."""
    d_i, _ = nurnberger.d(f, x_i, b, degree)

    # If the deviation on [x_i, b] is less than or equal to d_n, then no new knot can be placed in this interval
    if d_i <= d_n:
        return None

    # Set upper and lower bounds for bisection search
    x_l = x_i
    x_u = b

    for _ in range(max_iter):
        x_bar = (x_l + x_u) / 2

        # Compute the maximum deviation on [x_i, x_bar]
        d_i_max, approx = nurnberger.d(f, x_i, x_bar, degree)

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

    # Approximation on [x_i, x_max] is used to find the (degree + 2)-th alternance point
    approx = remez.remez(f, x_i, x_max, degree)
    x_min = approx.basis[degree + 1]

    return x_min, x_max


def discontinuous_spline(f, a, b, k, degree, tolerance=1e-6, max_iter=100):
    """Construct the Nurnberger free-knot spline approximation."""
    knots, d_min, d_max = nurnberger.step_zero(f, a, b, k, degree)

    x_min = None
    x_max = None
    d_n = None

    approximations = None

    for iteration in range(max_iter):
        if abs(d_max - d_min) < tolerance * d_max:
            break

        # target deviation for this iteration computed as geometric mean
        d_n = (d_min * d_max) ** 0.5

        new_knots = [a]
        new_approximations = []
        x_i = a

        # Subroutine: solve d(x_i, x_bar) = d_n while knots can be placed
        for i in range(k):
            x_min, x_max = subroutine(f, x_i, b, degree, d_n, max_iter)
            if x_min is None or x_max is None:
                break
            new_knots.append(x_min)

            # spline interval is [x_i, x_min].
            approx_i = remez.remez(f, x_i, x_min, degree)
            new_approximations.append(approx_i)

            x_i = x_min

        # Final interval [x_i, b].
        c_n, final_approx = nurnberger.d(f, x_i, b, degree)

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
            remez.remez(f, knots[i], knots[i + 1], degree)
            for i in range(len(knots) - 1)
        ]

    S = Spline.Spline(knots, [approx.g for approx in approximations])

    basis = [approx.basis for approx in approximations]
    approx = Spline.Approximation(f, S, (a, b), basis=basis)

    return approx, x_min, x_max


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 1  # number of free knots (not including a and b)
    m = 1  # degree of polynomial to fit

    approx, x_min, x_max = discontinuous_spline(f, a, b, k, m)

    _, t_star, d_star = approx.maxdeviation()

    print(f"Max deviation: {abs(d_star)}")
    print(f"knots: {approx.g.knots}")
    print(f"basis: {approx.basis}")
    print(f"x_min: {x_min}, x_max: {x_max}")

    plotting.plot_duo(
        approx,
        points=approx.basis,
        f_label=f_label,
        approximation_label="Piecewise polynomial approximation",
        points_label="Alternance points",
        title=f"Degree-{m} approximation with {len(approx.g.knots) - 2} free knots.",
        file_name=f"nberger_mod_{function_name}_a{a}_b{b}_k{k}_m{m}.png",
    )
