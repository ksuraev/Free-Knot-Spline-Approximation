# Original Nurnberger algorithm
import numpy as np

import plotting
import remez
import Spline
import test_functions

TOL = 1e-5


def d(f, a, b, m):
    """Return the maximum absolute deviation and Remez approximation on [a, b]."""
    approx = remez.run(f, a, b, m)
    _, _, d_max = approx.maxdeviation()

    return abs(d_max), approx


def step_zero(f, a, b, k, m):
    """Compute the initial knots and the maximum and minimum deviation over the interval [a,b]"""
    knots = np.linspace(a, b, k + 2)
    deviations = []

    for i in range(len(knots) - 1):
        d_i_max, _ = d(f, knots[i], knots[i + 1], m)
        deviations.append(d_i_max)

    d_min = min(deviations)
    d_max = max(deviations)

    return knots, d_min, d_max


def run(f, a, b, k, m, max_iter=100, verbose=False):
    """Construct the Nurnberger free-knot spline approximation."""
    knots, d_min, d_max = step_zero(f, a, b, k, m)

    # Track the approximations for each subinterval
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
        for z in range(k):
            d_i, _ = d(f, x_i, b, m)
            if d_i <= d_n:
                break

            # Set lower and upper bounds for x_bar
            x_l = x_i
            x_u = b

            # Use bisection method to find x_bar such that d(x_i, x_bar) = d_n
            for _ in range(max_iter):
                x_bar = (x_l + x_u) / 2
                d_i_max, approx_i = d(f, x_i, x_bar, m)

                if abs(d_i_max - d_n) < TOL * d_n:
                    break
                elif d_i_max < d_n:
                    x_l = x_bar
                else:
                    x_u = x_bar

            new_knots.append(x_bar)
            new_approximations.append(approx_i)
            x_i = x_bar

        # Final real interval
        c_n, final_approx = d(f, x_i, b, m)

        new_knots.append(b)
        new_approximations.append(final_approx)

        knots = np.array(new_knots)
        approximations = new_approximations

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    # If convergence happened before any new approximations were built
    if approximations is None:
        approximations = [
            remez.run(f, knots[i], knots[i + 1], m) for i in range(len(knots) - 1)
        ]

    # Create the final spline and approximation
    S = Spline.Spline(knots, [approx.g for approx in approximations])
    basis = [approx.basis for approx in approximations]
    approx = Spline.Approximation(f, S, (a, b), basis=basis)

    if verbose:
        _, t_star, d_star = approx.maxdeviation()
        print(f"Final max abs deviation: {abs(d_star):.5f} at t*={t_star:.5f}")
        print(f"Final knots: {S.knots}")
        print(f"Final basis: {basis}")

    return approx


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]
    a, b = test_functions.INTERVALS[function_name]

    # Report examples
    pairs = [(1, 2), (2, 1), (2, 2), (6, 2)]

    for k, m in pairs:
        approx = run(f, a, b, k, m)

        plotting.plot_single(
            approx,
            points=approx.basis,
            f_label=rf"{function_label}",
            approximation_label=rf"$s_{{{m}}}(t)$",
            title=rf"$k={k},\ m={m}$",
            file_name=f"nberger_{function_name}_k{k}_m{m}",
        )
