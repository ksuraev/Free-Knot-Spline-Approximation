import matplotlib.pyplot as plt
import numpy as np

import test_functions


def calculate_polynomial(f, xn, n):

    # Initialise matrix A and vector b
    A = np.zeros((n + 2, n + 2))
    b = np.zeros(n + 2)

    # Populate A and b
    for i in range(n + 2):
        x = xn[i]

        # Fill the polynomial terms 1, x, x^2, ..., x^n
        for j in range(n + 1):
            A[i, j] = x**j

        # Fill last column with the alternating error term (-1)^i
        A[i, n + 1] = (-1) ** i

        # Fill vector b with function values
        b[i] = f(x)

    # Solve the system Ax = b
    solution = np.linalg.solve(A, b)

    # Extract coefficients (first n+1 elements) and error (last element)
    coeffs = solution[:-1]
    E = solution[-1]

    # Create polynomial function from coefficients
    P = np.polynomial.Polynomial(coeffs)

    return P, E


def exchange(xn, x_new, d_max, errors):
    # If the new point is outside the leftmost point
    if x_new < xn[0]:
        if np.sign(d_max) == np.sign(errors[0]):
            # replace the leftmost point
            xn[0] = x_new
        else:
            # add x_new to the left and drop rightmost point
            xn = np.insert(xn, 0, x_new)[:-1]

    # If the new point is outside the rightmost point
    elif x_new > xn[-1]:
        if np.sign(d_max) == np.sign(errors[-1]):
            # replace the rightmost point
            xn[-1] = x_new
        else:
            # add x_new to the right and drop leftmost point
            xn = np.append(xn[1:], x_new)

    # If the new point is between two existing points
    else:
        for i in range(len(xn) - 1):
            if xn[i] < x_new < xn[i + 1]:

                # Replace with the closest point with same sign
                if np.sign(d_max) == np.sign(errors[i]):
                    xn[i] = x_new
                else:
                    xn[i + 1] = x_new
    return xn


def remez(f, a, b, n, tol=1e-6, max_iter=10000):
    # Guess initial n+2 points equidistantly spaced in the interval [a, b]
    xn = np.linspace(a, b, n + 2)

    for i in range(max_iter):
        P, E = calculate_polynomial(f, xn, n)

        # Discretise the interval into 10,000 points
        x_samples = np.linspace(a, b, 10000)
        d_samples = f(x_samples) - P(x_samples)

        # Find the absolute maximum error value (deviation)
        abs_errors = np.abs(d_samples)
        max_idx = np.argmax(abs_errors)

        x_max = x_samples[max_idx]
        d_max = d_samples[max_idx]

        # Check for convergence - Trefethen paper
        if abs(E) < 1e-14:
            converged = abs(d_max) < 1e-14
        else:
            converged = abs(d_max) - abs(E) <= tol * abs(E)
        if converged:
            return P, d_max, xn

        # Update the references points using single point exchange
        errors = f(xn) - P(xn)
        xn = exchange(xn, x_max, d_max, errors)

    return P, d_max, xn


def plot(f, f_label, P, xn, a, b, n, plot_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    # original function f(x)
    x = np.linspace(a, b, 1000)
    ax.plot(x, f(x), color="darkslategray", label=f_label)

    # approximation polynomial P(x)
    ax.plot(
        x,
        P(x),
        linewidth=2,
        color="cornflowerblue",
        label=rf"$P_{{{n}}}(t)$",
    )

    # alternance points
    for x_i in xn:
        y_f = f(x_i)
        y_p = P(x_i)

        # point on P
        ax.scatter(x_i, y_p, color="black", s=30, zorder=5)

        # deviation between f and P
        ax.plot(
            [x_i, x_i],
            [y_p, y_f],
            color="black",
            linestyle=(0, (8, 5)),
            linewidth=0.8,
            alpha=0.7,
        )
    # ax.set_title(f"Remez degree {n} approximation.")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=15, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(plot_name, dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    m = 50
    P, d_max, xn = remez(f, a, b, m)
    print(f"Max error: {d_max}")
    print(f"Alternance points: {xn}")
    plot(f, f_label, P, xn, a, b, m, f"remez_{function_name}_m{m}_a{a}_b{b}.png")
