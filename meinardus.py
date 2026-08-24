import matplotlib.pyplot as plt
import numpy as np

import remez


def d(f, a, b, degree):
    """Compute the maximum deviation over the interval [a,b] using the Remez algorithm. Returns the maximum deviation and the alternance points."""
    _, d_max, alt_pts = remez.remez(f, a, b, degree)
    return abs(d_max), alt_pts


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


def run(f, a, b, k, degree, use_last_alt_pt, tolerance=1e-6, max_iter=100):
    """Run the Meinardus algorithm to find the optimal placement of k free knots in the interval [a,b] for polynomial approximation of degree 'degree'."""
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
        for _ in range(k):
            d_i, _ = d(f, x_i, b, degree)
            if d_i <= d_n:
                break

            # Set lower and upper bounds for x_bar
            x_l = x_i
            x_u = b

            for _ in range(max_iter):
                x_bar = (x_l + x_u) / 2
                d_i_max, alt_pts = d(f, x_i, x_bar, degree)

                if abs(d_i_max - d_n) < 1e-10 * d_n:
                    break
                elif d_i_max < d_n:
                    x_l = x_bar
                else:
                    x_u = x_bar

            if use_last_alt_pt:
                new_knot = alt_pts[-1]
                new_knots.append(new_knot)
                x_i = new_knot
            else:
                new_knots.append(x_bar)
                x_i = x_bar

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

    return knots, d_min, d_max


def plot(f, knots, degree, a, b, plot_name):
    piecewise_polynomial = []
    for i in range(len(knots) - 1):
        P, _, _ = remez.remez(f, knots[i], knots[i + 1], degree)
        piecewise_polynomial.append((knots[i], knots[i + 1], P))

    fig, ax = plt.subplots(figsize=(10, 6))

    # original function f(x)
    x = np.arange(a, b, 0.01)
    ax.plot(x, f(x), color="slategray", label="f(x)")

    # approximation polynomial P(x) over each interval
    for start, end, P in piecewise_polynomial:
        x = np.arange(start, end, 0.01)
        ax.plot(
            x, P(x), color="dodgerblue", label="P(x)" if start == knots[0] else None
        )

    for x in knots:
        ax.axvline(x, color="red", ls=":", label="knots" if x == knots[0] else None)

    ax.set_title(f"Degree-{degree} approximation with {k} free knots")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_name)
    plt.show()


if __name__ == "__main__":

    def f(x):
        return np.sin(x)

    a, b = 0, 6
    k = 2  # number of free knots (not including a and b)
    degree = 1  # degree of polynomial to fit

    knots, d_min, d_max = run(f, a, b, k, degree, False)
    print(f"x_bar knots:       {knots}. Max and min d: {d_max:.8f}, {d_min:.8f}")
    # plot(knots, degree, a, b, "xbar_knots.png")

    knots, d_min, d_max = run(f, a, b, k, degree, True)
    print(f"Last alt pt knots: {knots}. Max and min d: {d_max:.8f}, {d_min:.8f}")
    # plot(knots, degree, a, b, "last_alt_knots.png")
