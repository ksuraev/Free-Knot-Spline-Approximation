import matplotlib.pyplot as plt
import numpy as np


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


def exchange(xn, x_new, e_max, errors):
    # If the new point is outside the leftmost point
    if x_new < xn[0]:
        if np.sign(e_max) == np.sign(errors[0]):
            # replace the leftmost point
            xn[0] = x_new
        else:
            # add x_new to the left and drop rightmost point
            xn = np.insert(xn, 0, x_new)[:-1]

    # If the new point is outside the rightmost point
    elif x_new > xn[-1]:
        if np.sign(e_max) == np.sign(errors[-1]):
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
                if np.sign(e_max) == np.sign(errors[i]):
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
        e_samples = f(x_samples) - P(x_samples)

        # Find the absolute maximum error value (deviation)
        abs_errors = np.abs(e_samples)
        max_idx = np.argmax(abs_errors)

        x_max = x_samples[max_idx]
        e_max = e_samples[max_idx]

        # Check for convergence - Trefethen paper
        if abs(E) < 1e-14:
            converged = abs(e_max) < 1e-14
        else:
            C = abs(e_max) / abs(E)
            converged = C <= 1 + tol
        if converged:
            return P, e_max, xn

        # Update the references points using single point exchange
        errors = f(xn) - P(xn)
        xn = exchange(xn, x_max, e_max, errors)

    return P, e_max, xn


def plot(f, P, xn, a, b, n, plot_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    # original function f(x)
    x = np.linspace(a, b, 1000)
    ax.plot(x, f(x), color="slategray", label="f(x)")

    # approximation polynomial P(x)
    ax.plot(x, P(x), color="dodgerblue", label="P(x)")

    # alternance points
    for x in xn:
        ax.axvline(x=x, color="lightgray", linestyle="--", alpha=0.5)

    ax.set_title(f"Remez approximation of degree {n}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_name)
    plt.show()


if __name__ == "__main__":

    def f(x):
        return np.sin(x)

    a, b = 0, 4
    n = 2
    P, e_max, xn = remez(f, a, b, n)
    print(f"Max error: {e_max}")
    print(f"Alternance points: {xn}")
    # plot(f, P, xn, a, b, n, "remez.png")
