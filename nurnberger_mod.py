import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import remez
import test_functions

ALTERNANCE_TOL = 1e-4


def d(f, a, b, degree):
    """Compute the maximum deviation over the interval [a,b] using the Remez algorithm. Returns the maximum deviation and the alternance points."""
    _, d_max, alt_pts = remez.remez(f, a, b, degree)
    return abs(d_max), alt_pts


def step_zero(f, a, b, k, degree):
    """Compute the initial knots and the maximum and minimum deviation over the interval [a,b]"""
    knots = np.linspace(a, b, k + 2)
    # knots = [a, 3 * np.pi, b]
    deviations = []

    for i in range(len(knots) - 1):
        d_i_max, _ = d(f, knots[i], knots[i + 1], degree)
        deviations.append(d_i_max)

    d_min = min(deviations)
    d_max = max(deviations)

    return knots, d_min, d_max


def deviation(f, P, t):
    return f(t) - P(t)


def find_local_abs_deviation_maxima(f, P, a, b, n_samples=10000):
    t_samples = np.linspace(a, b, n_samples)
    d_samples = deviation(f, P, t_samples)

    abs_d_samples = np.abs(d_samples)

    indices = []

    if abs_d_samples[0] >= abs_d_samples[1]:
        indices.append(0)

    for j in range(1, len(t_samples) - 1):
        if (
            abs_d_samples[j] >= abs_d_samples[j - 1]
            and abs_d_samples[j] >= abs_d_samples[j + 1]
        ):
            indices.append(j)

    if abs_d_samples[-1] >= abs_d_samples[-2]:
        indices.append(len(t_samples) - 1)

    return [(t_samples[j], d_samples[j]) for j in indices]


def find_alternance_points(f, P, a, b):
    extrema = find_local_abs_deviation_maxima(f, P, a, b)
    if not extrema:
        return np.array([])

    global_max = max(abs(d) for _, d in extrema)

    filtered = [
        (t, d) for t, d in extrema if abs(abs(d) - global_max) <= ALTERNANCE_TOL
    ]

    unique = []

    for t, dev in filtered:
        if not unique or not np.isclose(t, unique[-1][0]):
            unique.append((t, dev))

    return np.array([t for t, _ in unique])


def subroutine(f, function_name, x_i, b, degree, d_n, max_iter):
    results = []

    d_i, _ = d(f, x_i, b, degree)

    if d_i <= d_n:
        return

    # Set lower and upper bounds for x_bar
    x_l = x_i
    x_u = b

    for _ in range(max_iter):
        x_bar = (x_l + x_u) / 2
        d_i_max, alt_pts = d(f, x_i, x_bar, degree)

        if x_u - x_l < 1e-5:
            break
        elif d_i_max <= d_n + 1e-5:
            x_l = x_bar
        else:
            x_u = x_bar

        results.append(
            {
                "d_n": d_n,
                "d_i_max": d_i_max,
                "x_l": x_l,
                "x_bar": x_bar,
                "x_u": x_u,
                # "x_l_dev": deviation(f, remez.remez(f, x_i, x_l, degree)[0], x_l),
                # "x_bar_dev": deviation(f, remez.remez(f, x_i, x_bar, degree)[0], x_bar),
                # "x_u_dev": deviation(f, remez.remez(f, x_i, x_u, degree)[0], x_u),
            }
        )

    x_max = x_l

    _, alt_pts_remez = d(f, x_i, x_max, degree)
    P, _, _ = remez.remez(f, x_i, x_max, degree)
    alt_pts_find = find_alternance_points(f, P, x_i, x_max)

    # print(f"Alt points from remez(): {alt_pts_remez}")
    # print(f"alt points from find() : {alt_pts_find}")

    x_min = alt_pts_find[degree + 1]

    # for result in results:
    #     result["x_min"] = x_min
    #     result["x_max"] = x_max
    #     result["alternance_points_remez"] = alt_pts_remez
    #     result["alternance_points_find"] = alt_pts_find

    # with open(f"n_{function_name}_results.csv", "a", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=results[0].keys())
    #     if file.tell() == 0:
    #         writer.writeheader()
    #     writer.writerows(results)

    if x_max != x_min:
        print(f"x_min: {x_min}, x_max: {x_max}")

    return x_min, x_max, alt_pts_find


def find_max_deviation_overall(f, polynomial):
    all_extrema = []

    for i, (start, end, P) in enumerate(polynomial):
        extrema = find_local_abs_deviation_maxima(
            f,
            P,
            start,
            end,
        )

        for t, dev in extrema:
            all_extrema.append((i, t, dev))

    i_star, t_star, d_star = max(
        all_extrema,
        key=lambda item: abs(item[2]),
    )

    return i_star, t_star, d_star


def discontinuous_spline(
    f, function_name, a, b, k, degree, tolerance=1e-6, max_iter=10000
):
    """Run the Nurnberger algorithm to find the optimal placement of k free knots in the interval [a,b] for polynomial approximation of degree 'degree'."""
    knots, d_min, d_max = step_zero(f, a, b, k, degree)

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
        c_n, _ = d(f, knots[j], knots[j + 1], degree)

        # Update d_min and d_max for the next iteration
        d_min = max(d_min, min(c_n, d_n))
        d_max = min(d_max, max(c_n, d_n))

    return knots, x_min, x_max, d_n


def plot(f, f_label, polynomial, a, b, knots, alt_pts, m, k, file_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    t = np.linspace(a, b, 1000)
    ax1.plot(t, f(t), color="slategrey", label=f_label)

    for i, (start, end, P) in enumerate(polynomial):
        t_interval = np.linspace(start, end, 1000)

        ax1.plot(
            t_interval,
            P(t_interval),
            color="dodgerblue",
            label="P(t)" if i == 0 else None,
        )
        ax2.plot(
            t_interval,
            deviation(f, P, t_interval),
            color="dodgerblue",
        )

    # knots
    knot_label = ", ".join(f"{knot:g}" for knot in knots)

    for j, knot in enumerate(knots):
        for ax in (ax1, ax2):
            ax.axvline(
                knot,
                color="red",
                linestyle="-",
                label=f"Knots: {knot_label}" if j == 0 else None,
            )

    # alternance points
    alt_pts_label = ", ".join(f"{point:g}" for point in alt_pts)

    for j, point in enumerate(alt_pts):
        for ax in (ax1, ax2):
            ax.axvline(
                point,
                linestyle=":",
                color="black",
                label=f"Alternance points: {alt_pts_label}" if j == 0 else None,
            )

    ax1.set_xlabel("t")
    ax1.set_title("Polynomial approximation")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    ax2.axhline(0)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$f(t)-P(t)$")
    ax2.set_title("Deviation")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    fig.suptitle(f"Degree-{m} approximation of {f_label} with {k} internal knots")

    fig.tight_layout()
    fig.savefig(file_name)
    plt.show()


if __name__ == "__main__":

    function_name = "cos_weird"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    file_name = f"n_{function_name}_results.csv"
    with open(file_name, "w", newline=""):
        pass

    a, b = 0, 12
    k = 1  # number of free knots (not including a and b)
    m = 1  # degree of polynomial to fit

    knots, x_min, x_max, d_n = discontinuous_spline(f, function_name, a, b, k, m)

    all_alt_pts = []
    polynomial = []
    for i in range(len(knots) - 1):
        P, _, _ = remez.remez(f, knots[i], knots[i + 1], m)
        polynomial.append((knots[i], knots[i + 1], P))
        alt_pts = find_alternance_points(f, P, knots[i], knots[i + 1])
        all_alt_pts.extend(alt_pts)

    print("all alt pts:", all_alt_pts)

    plot(
        f,
        f_label,
        polynomial,
        a,
        b,
        knots,
        all_alt_pts,
        m,
        k,
        f"n_{function_name}_a{a}_b{b}_knot{knots[1]:.5f}_k{k}_m{m}.png",
    )
    i_star, t_star, d_star = find_max_deviation_overall(f, polynomial)

    print(f"Max deviation: i_star={i_star}, t_star={t_star}, d_star={d_star}")

    # print(f"deviation at 0: {deviation(f, polynomial[0][2], 0)}")

    # pts = [
    #     3.14191269,
    #     6.28262527,
    #     7.33032954,
    #     8.37803381,
    #     9.42453796,
    #     10.47224223,
    #     11.51874638,
    # ]
    # for pt in pts:
    #     for start, end, P in polynomial:
    #         if start <= pt <= end:
    #             print(f"Deviation at {pt}: {deviation(f, P, pt)}")
    #             break

    # knots, d_min, d_max = run(f, a, b, k, m, True)
    # print(f"Last alt pt knots: {knots}. Max and min d: {d_max:.8f}, {d_min:.8f}")
    # plot(knots, m, a, b, "last_alt_knots.png")
