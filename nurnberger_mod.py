import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import remez
import test_functions

ALTERNANCE_TOL = 1e-5


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
    d_samples = np.array([deviation(f, P, t) for t in t_samples])

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


def find_alternance_points_in_interval(f, P, a, b):
    extrema = find_local_abs_deviation_maxima(f, P, a, b)
    if not extrema:
        return np.array([]), np.array([])

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

    # Set lower and upper bounds for x_max
    x_l = x_i
    x_u = b

    for _ in range(max_iter):
        x_bar = (x_l + x_u) / 2
        d_i_max, alt_pts = d(f, x_i, x_bar, degree)

        if x_u - x_l < 1e-5:
            break
        elif d_i_max <= d_n + 1e-6:
            x_l = x_bar
        else:
            x_u = x_bar
        results.append(
            {
                "x_i": x_i,
                "b": b,
                "d_n": d_n,
                "x_l": x_l,
                "x_u": x_u,
                "x_bar": x_bar,
                "d_i_max": d_i_max,
            }
        )

    x_max = x_l
    d_at_x_max, alt_pts_remez = d(f, x_i, x_max, degree)
    P, _, _ = remez.remez(f, x_i, x_max, degree)
    alt_pts_find = find_alternance_points_in_interval(f, P, x_i, x_max)

    print(f"Alt points from remez() in interval [{x_i, x_max}]: {alt_pts_remez}")
    print(f"alt points from find() in interval [{x_i, x_max}] : {alt_pts_find}")

    x_min = alt_pts_find[degree + 1]

    for result in results:
        result["x_max"] = x_max
        result["x_min"] = x_min
        result["alternance_points"]: alt_pts_find

    with open(f"n_{function_name}_results.csv", "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    if x_max != x_min:
        print(f"x_min: {x_min}, x_max: {x_max}")

    return x_min, x_max, alt_pts_find


def run(f, function_name, a, b, k, degree, tolerance=1e-6, max_iter=10000):
    """Run the Nurnberger algorithm to find the optimal placement of k free knots in the interval [a,b] for polynomial approximation of degree 'degree'."""
    knots, d_min, d_max = step_zero(f, a, b, k, degree)

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
            x_min, x_max, alt_pts = subroutine(
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

    return knots, d_min, d_max, alt_pts


# find the overall maximum and minimum deviation across all intervals
def find_extrema_overall(f, P, knots, n, n_samples=10000):
    t_max = None
    d_max = -np.inf
    i_max = None

    t_min = None
    d_min = np.inf
    i_min = None

    for i in range(n):
        if i == 0:
            t_samples = np.linspace(knots[i], knots[i + 1], n_samples)
        else:
            t_samples = np.linspace(knots[i], knots[i + 1], n_samples)[1:]

        d_samples = np.array([deviation(f, P, t) for t in t_samples])

        idx_max = np.argmax(d_samples)
        idx_min = np.argmin(d_samples)

        # max signed deviation
        if d_samples[idx_max] > d_max:
            i_max, t_max, d_max = i, t_samples[idx_max], d_samples[idx_max]

        # min signed deviation
        if d_samples[idx_min] < d_min:
            i_min, t_min, d_min = i, t_samples[idx_min], d_samples[idx_min]

    # max absolute deviation
    if abs(d_max) >= abs(d_min):
        i_star, t_star, d_star = i_max, t_max, d_max
    else:
        i_star, t_star, d_star = i_min, t_min, d_min

    return (
        (i_max, t_max, d_max),
        (i_min, t_min, d_min),
        (i_star, t_star, d_star),
    )


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

    function_name = "cos_if_else"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    file_name = f"n_{function_name}_results.csv"
    with open(file_name, "w", newline=""):
        pass

    a, b = 0, 12
    k = 1  # number of free knots (not including a and b)
    m = 1  # degree of polynomial to fit

    knots, d_min, d_max, alt_pts = run(f, function_name, a, b, k, m)

    polynomial = []
    all_alt_pts = []
    for i in range(len(knots) - 1):
        P, _, _ = remez.remez(f, knots[i], knots[i + 1], m)
        polynomial.append((knots[i], knots[i + 1], P))
        alt_pts = find_alternance_points_in_interval(f, P, knots[i], knots[i + 1])
        all_alt_pts.extend(alt_pts)

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
        f"n_{function_name}_k{k}_m{m}.png.png",
    )

    # knots, d_min, d_max = run(f, a, b, k, m, True)
    # print(f"Last alt pt knots: {knots}. Max and min d: {d_max:.8f}, {d_min:.8f}")
    # plot(knots, m, a, b, "last_alt_knots.png")
