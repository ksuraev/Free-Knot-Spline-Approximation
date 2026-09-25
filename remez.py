import numpy as np

import plotting
import Spline
import test_functions

CONVERGENCE_TOL = 1e-14
DEVIATION_TOL = 1e-6


def calculate_polynomial(f, basis, m):
    """Calculate the polynomial approximation of degree m to function f using the given basis points."""
    # Initialise matrix A and vector b
    A = np.zeros((m + 2, m + 2))
    b = np.zeros(m + 2)

    # Populate A and b
    for i in range(m + 2):
        x = basis[i]

        # Fill the polynomial terms 1, x, x^2, ..., x^m
        for j in range(m + 1):
            A[i, j] = x**j

        # Fill last column with the alternating deviation term (-1)^i
        A[i, m + 1] = (-1) ** i

        # Fill vector b with function values
        b[i] = f(x)

    # Solve the system Ax = b
    solution = np.linalg.solve(A, b)

    # Extract coefficients (first m+1 elements) and deviation (last element)
    coeffs = solution[:-1]
    E = solution[-1]

    # Create polynomial function from coefficients
    P = Spline.Polynomial(coeffs)

    return P, E


def exchange(basis, t_new, d_max, errors):
    """VP basis exchange algorithm to update the basis points with a new point t_new."""
    # If the new point is outside the leftmost point
    if t_new < basis[0]:
        if np.sign(d_max) == np.sign(errors[0]):
            # replace the leftmost point
            basis[0] = t_new
        else:
            # add t_new to the left and drop rightmost point
            basis = np.insert(basis, 0, t_new)[:-1]

    # If the new point is outside the rightmost point
    elif t_new > basis[-1]:
        if np.sign(d_max) == np.sign(errors[-1]):
            # replace the rightmost point
            basis[-1] = t_new
        else:
            # add t_new to the right and drop leftmost point
            basis = np.append(basis[1:], t_new)

    # If the new point is between two existing points
    else:
        for i in range(len(basis) - 1):
            if basis[i] < t_new < basis[i + 1]:
                # Replace with the closest point with same sign
                if np.sign(d_max) == np.sign(errors[i]):
                    basis[i] = t_new
                else:
                    basis[i + 1] = t_new
    return basis


def run(f, a, b, m, max_iter=10000, verbose=False):
    """Remez algorithm for polynomial approximation of degree m to function f on interval [a, b]."""
    # Guess initial m+2 points equidistantly spaced in the interval [a, b]
    basis = np.linspace(a, b, m + 2)

    for i in range(max_iter):
        P, E = calculate_polynomial(f, basis, m)
        approx = Spline.Approximation(f, P, [a, b], basis)

        # Find maximum absolute deviation over [a, b]
        _, t_max, d_max = approx.maxdeviation()

        # Check for convergence - Trefethen paper
        if abs(E) < CONVERGENCE_TOL:
            converged = abs(d_max) < CONVERGENCE_TOL
        else:
            converged = abs(d_max) - abs(E) <= DEVIATION_TOL * abs(E)
        if converged:
            if verbose:
                print(f"Final max abs deviation: {abs(d_max):.5f} at t*={t_max:.5f}")
                print(f"Final basis: {approx.basis}")
            return approx

        # Update the basis points using VP exchange
        basis_dev = f(basis) - P(basis)
        basis = exchange(basis, t_max, d_max, basis_dev)

    approx = Spline.Approximation(f, P, [a, b], basis)

    if verbose:
        _, t_star, d_star = approx.maxdeviation()
        print(f"Final max abs deviation: {abs(d_star):.5f} at t*={t_star:.5f}")
        print(f"Final basis: {approx.basis}")

    return approx


if __name__ == "__main__":
    # Example usage matching report examples
    function_name = "g"
    f, _ = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]
    a, b = test_functions.INTERVALS[function_name]

    for m in [2, 5, 12, 30]:
        approx = run(f, a, b, m)

        plotting.plot_single(
            approx,
            points=approx.basis,
            f_label=rf"{function_label}",
            approximation_label=rf"$P_{{{m}}}(t)$",
            title=rf"$m={m}$",
            file_name=f"remez_{function_name}_m{m}",
        )
