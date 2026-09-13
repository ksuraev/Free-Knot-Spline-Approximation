import numpy as np

import helper
import nurnberger
import plotting
import remez
import test_functions

ALTERNANCE_TOL = 1e-4
TOL = 1e-5


def deviation(f, P, t):
    return f(t) - P(t)


def subroutine(f, function_name, x_i, b, degree, d_n, max_iter):
    # Compute abs max deviation in the interval [x_i, b]
    d_i, _ = nurnberger.d(f, x_i, b, degree)

    # If the deviation is already less than or equal to d_n, no need to find x_bar
    if d_i <= d_n:
        return

    # Set lower and upper bounds for x_bar
    x_l = x_i
    x_u = b

    for _ in range(max_iter):
        # Bisect the interval
        x_bar = (x_l + x_u) / 2

        # Compute the maximum deviation in the interval [x_i, x_bar]
        d_i_max, alt_pts = nurnberger.d(f, x_i, x_bar, degree)

        if x_u - x_l < TOL:
            break
        elif d_i_max <= d_n + TOL:
            x_l = x_bar
        else:
            x_u = x_bar

    # After the loop, x_l is the largest x_bar such that d_i_max <= d_n
    x_max = x_l

    # Compute the polynomial approximation in the interval [x_i, x_max]
    P, _, _ = remez.remez(f, x_i, x_max, degree)

    # Find the alternance points in the interval [x_i, x_max]
    all_alt_pts, signs, _ = helper.find_alternance_points(
        lambda i, t: deviation(f, P, t), [x_i, x_max], tol=ALTERNANCE_TOL
    )

    # x_min is the (m+2)-nd alternance point
    x_min = all_alt_pts[degree + 1]

    return x_min, x_max, all_alt_pts


def discontinuous_spline(
    f, function_name, a, b, k, degree, tolerance=1e-6, max_iter=10000
):
    """Run the Nurnberger algorithm to find the optimal placement of k free knots in the interval [a,b] for polynomial approximation of degree 'degree'."""
    knots, d_min, d_max = nurnberger.step_zero(f, a, b, k, degree)

    x_min = None
    x_max = None
    d_n = None

    for iteration in range(max_iter):
        if abs(d_max - d_min) < tolerance * d_max:
            break

        # target deviation for this iteration computed as geometric mean
        d_n = (d_min * d_max) ** 0.5

        new_knots = [a]
        x_i = a
        j = 0

        # Subroutine: solve d(x_i, x_bar) = d_n while knots can be placed
        for i in range(k):
            x_min, x_max, _ = subroutine(
                f, function_name, x_i, b, degree, d_n, max_iter
            )
            new_knots.append(x_min)
            x_i = x_min
            j += 1

        # Collapse any unplaced knots to the right endpoint b
        remaining = k - j
        new_knots.extend([b] * remaining)
        new_knots.append(b)
        knots = np.array(new_knots)

        # c_n = deviation of the last real interval
        c_n, _ = nurnberger.d(f, knots[j], knots[j + 1], degree)

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    return knots, x_min, x_max, d_n


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 1  # number of free knots (not including a and b)
    m = 1  # degree of polynomial to fit

    knots, x_min, x_max, d_n = discontinuous_spline(f, function_name, a, b, k, m)

    polynomials = []
    alt_pts = []

    # todo: nurnberger already calls remez, so should return polynomials from discontinuous_spline (ideally)
    for i in range(len(knots) - 1):
        P, _, alt = remez.remez(f, knots[i], knots[i + 1], m)
        polynomials.append(P)
        alt_pts.extend(alt)

    # For plotting
    def P(i, t):
        return polynomials[i](t)

    # Find the maximum deviation over all intervals
    _, _, (i_star, t_star, d_star) = helper.find_extrema_overall(
        lambda i, t: f(t) - P(i, t), knots
    )

    print(f"Max deviation: i_star={i_star}, t_star={t_star}, d_star={d_star}")

    plotting.plot_detailed(
        f,
        P,
        a,
        b,
        knots=knots,
        points=alt_pts,
        f_label=f_label,
        approximation_label="Piecewise polynomial approximation",
        points_label="Alternance points",
        title=f"Degree-{m} approximation with {len(knots) - 2} free knots. Max abs deviation: {d_star:.5f}.",
        file_name=f"nberger_{function_name}_a{a}_b{b}_k{k}_m{m}.png",
    )

    # plotting.plot_report(
    #     f,
    #     P,
    #     a,
    #     b,
    #     knots=knots,
    #     points=alt_pts,
    #     f_label=f_label,
    #     approximation_label=rf"$S_{{{m}}}(t)$",
    #     points_label="",
    #     file_name=f"r_nberger_{function_name}_a{a}_b{b}_k{k}_m{m}.png",
    # )
