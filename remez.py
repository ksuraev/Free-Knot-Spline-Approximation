import matplotlib.pyplot as plt
import numpy as np


def calculate_polynomial(f: callable, xn: list, n: int):

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


def exchange(xn: list, x_new: float, e_max: float, errors: list):
    # If the new point is outside the leftmost point
    if x_new < xn[0]:
        if np.sign(e_max) == np.sign(errors[0]):
            xn[0] = x_new  # replace the leftmost point
        else:
            xn = np.insert(xn, 0, x_new)[
                :-1
            ]  # add new point to the left and drop rightmost point
    # If the new point is outside the rightmost point
    elif x_new > xn[-1]:
        if np.sign(e_max) == np.sign(errors[-1]):
            xn[-1] = x_new  # replace the rightmost point
        else:
            xn = np.append(
                xn[1:], x_new
            )  # add new point to the right and drop leftmost point
    else:
        # If the new point is between two existing points
        for i in range(len(xn) - 1):
            if xn[i] < x_new < xn[i + 1]:

                # Replace with the closest point with same sign
                if np.sign(e_max) == np.sign(errors[i]):
                    xn[i] = x_new
                else:
                    xn[i + 1] = x_new


def remez(
    f: callable, a: float, b: float, n: int, tol: float = 1e-4, max_iter: int = 100
):
    # Guess initial n+2 points equidistantly spaced in the interval [a, b]
    xn = np.linspace(a, b, n + 2)

    P = None
    e_max = None

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
        C = abs(e_max) / abs(E)
        if C <= 1 + tol:
            return P, e_max

        # Update the references points using single point exchange
        errors = f(xn) - P(xn)
        exchange(xn, x_max, e_max, errors)

    return P, e_max
