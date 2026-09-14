# Original Nurnberger algorithm
import numpy as np

import plotting
import remez
import Spline
import test_functions

TOL = 1e-5


def d(f, a, b, degree):
    """Return the maximum absolute deviation and Remez approximation on [a, b]."""
    approx = remez.remez(f, a, b, degree)
    _, _, d_max = approx.maxdeviation()

    return abs(d_max), approx


def step_zero(f, a, b, k, degree):
    """Compute the initial knots and the maximum and minimum deviation over the interval [a,b]"""
    knots = np.linspace(a, b, k + 2)
    deviations = []

    for i in range(len(knots) - 1):
        d_i_max, _ = d(f, knots[i], knots[i + 1], degree)
        deviations.append(d_i_max)

    d_min = min(deviations)
    d_max = max(deviations)

    return knots, d_min, d_max


def run(f, a, b, k, degree, tolerance=1e-6, max_iter=100):
    """Run the Nurnberger algorithm to find the optimal placement of k free knots in the interval [a,b] for polynomial approximation of degree 'degree'."""
    knots, d_min, d_max = step_zero(f, a, b, k, degree)

    # Track the approximations for each subinterval
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
        for z in range(k):
            d_i, _ = d(f, x_i, b, degree)
            if d_i <= d_n:
                break

            # Set lower and upper bounds for x_bar
            x_l = x_i
            x_u = b

            for _ in range(max_iter):
                x_bar = (x_l + x_u) / 2
                d_i_max, approx_i = d(f, x_i, x_bar, degree)

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
        c_n, final_approx = d(f, x_i, b, degree)

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
            remez.remez(f, knots[i], knots[i + 1], degree)
            for i in range(len(knots) - 1)
        ]

    S = Spline.Spline(knots, [approx.g for approx in approximations])

    basis = [approx.basis for approx in approximations]

    return Spline.Approximation(f, S, (a, b), basis=basis)


if __name__ == "__main__":
    function_name = "g"

    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 1
    degree = 1

    approx = run(f, a, b, k, degree)

    _, t_star, d_star = approx.maxdeviation()

    print(f"Max deviation: {abs(d_star)}")
    print(f"knots: {approx.g.knots}")
    print(f"basis: {approx.basis}")

    plotting.plot_duo(
        approx,
        points=approx.basis,
        f_label=f_label,
        approximation_label="Piecewise polynomial approximation",
        points_label="Alternance points",
        title=f"Degree-{degree} approximation with {len(approx.g.knots) - 2} free knots.",
        file_name=f"nberger_{function_name}_a{a}_b{b}_k{k}_m{degree}.png",
    )
