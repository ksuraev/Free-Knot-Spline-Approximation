import warnings
from pathlib import Path

import numpy as np
import pulp as pl
import qpsolvers
from qpsolvers.conversions.ensure_sparse_matrices import SparseConversionWarning

import nadia_original
import nurnberger_mod
import plotting
import Spline
import test_functions

warnings.filterwarnings("ignore", category=SparseConversionWarning)


def subgradients(basis, S, signs):
    """Compute the subgradients"""
    P = nadia_original.build_P(basis, S.knots, S.degree)
    P = P.T

    M = np.zeros((len(S.knots) - 2, P.shape[1]))
    a = [p.coef[1:] for p in S.polynomials]
    for i, knot in enumerate(S.knots[1:-1]):
        col = 0
        for t in basis:
            if t > knot:
                M[i, col] = sum(
                    -(j + 1) * a[i][j] * (t - knot) ** j for j in range(S.degree)
                )

            col += 1

    G = np.concatenate([np.ones((1, P.shape[1])), P, M], axis=0)
    G = np.multiply(G, signs)

    return G


def descent_direction(basis, S, signs):
    """Solve the QP to find the descent direction for Psi bar"""
    G = subgradients(basis, S, signs)
    P = G.T @ G
    q = np.zeros(G.shape[1])
    A = np.ones(G.shape[1])
    b = np.ones(1)

    problem = qpsolvers.Problem(P, q, None, None, A, b, lb=q)
    solution = qpsolvers.solve_problem(problem, solver="proxqp")

    x = solution.x
    v = G @ x

    return -v


def simplex_system(f, samples, knots, m):
    P = nadia_original.build_P(samples, knots, m)
    M = np.concatenate([np.ones((len(samples), 1)), P], axis=1)

    # Stack M and -M vertically
    A = np.concatenate([M, -M], axis=0)

    # Stack f(samples) and -f(samples) vertically to create b
    b = np.concatenate([f(samples), -f(samples)], axis=0)

    return A, b


def Psi_bar(f, knots, m):
    """Evaluate Psi bar at the given knots and return the approximation and signs of the deviation."""
    nsamples = 1000
    samples = np.linspace(knots[0], knots[-1], nsamples)
    A, b = simplex_system(f, samples, knots, m)

    # Objective function: minimise the last variable (the deviation)
    c = np.zeros(A.shape[1])
    c[-1] = 1

    prob = pl.LpProblem("simplex", pl.LpMinimize)

    # Create variables for the coefficients and the deviation
    x = pl.LpVariable.dicts("x_%s", range(0, A.shape[1]), lowBound=None)
    z = pl.LpVariable("z", lowBound=None)

    # Add the deviation variable to the objective
    prob += z

    # Loop over constraint rows
    for i in range(A.shape[0]):
        constraint = (
            pl.lpSum(A[i, j] * x[j] for j in range(A.shape[1])) - z <= b[i],
            f"{i}",
        )
        prob += constraint

    # Solve using the HiGHS
    solver = pl.HiGHS(msg=False)
    prob.solve(solver)

    # The active constraints should give us the maximum deviation points:
    active = [int(name) for name, c in list(prob.constraints.items()) if c.slack == 0]
    t_active = np.array(
        [
            [samples[i], -1] if i < nsamples else [samples[i - nsamples], 1]
            for i in active
        ]
    )

    # Sort the active points by their t values and extract the basis and signs
    t_active = t_active[np.argsort(t_active[:, 0])]
    basis = t_active[:, 0]  # [t for [t, _] in t_active]
    signs = t_active[:, 1]  # [s for [_, s] in t_active]

    # Extract the solution for the coefficients and the deviation
    r = np.array([x[j].varValue for j in range(A.shape[1])])
    a = np.concatenate([r[1:].reshape(len(knots) - 1, m)])
    coeffs = [np.concatenate([[0], c]) for c in a]
    coeffs[0][0] = r[0]

    # Create the spline from the coefficients and knots
    polynomials = [Spline.Polynomial(c, offset=x) for c, x in zip(coeffs, knots[:-1])]
    S = Spline.SUSpline(knots, polynomials)
    A = Spline.Approximation(f, S, [knots[0], knots[-1]], basis=basis)

    return z.varValue, A, signs


def armijo_line_search(f, knots, m, d, rho=0.5, c=0.1):
    """Armijo line search to find the next theta in the direction of d."""
    # Initialise the step size
    alpha = 100.0
    d_full = np.concatenate([[0], d, [0]])
    curr_knots = knots.copy()
    psi_theta, _, _ = Psi_bar(f, curr_knots, m)

    while alpha > 1e-8:
        # Compute the next knots using the step size alpha
        next_knots = curr_knots + alpha * d_full

        # Check if the new knots are valid
        if np.any(next_knots[:-1] >= next_knots[1:]):
            alpha *= rho
            continue

        psi_next, _, _ = Psi_bar(f, next_knots, m)

        # Check the Armijo condition
        if psi_next <= psi_theta + c * alpha * (np.linalg.norm(d) ** 2):
            return next_knots

        # If the Armijo condition is not satisfied, reduce alpha and try again
        alpha *= rho

    return curr_knots


def evaluate_and_get_direction(f, knots, m):
    """Evaluate Psi bar at the given knots and return the descent direction, spline, approximation, and deviation."""
    # Evaluate Psi bar at the given knots
    deviation, approx, signs = Psi_bar(f, knots, m)

    # Compute the descent direction using the subgradients and the simplex system
    S = approx.g
    basis = approx.basis
    d = descent_direction(basis, S, signs)

    return d, S, approx, deviation


def descent_algorithm(theta_start, f, a, b, m, k, track_iterates=False):
    """Descent algorithm to find the optimal internal knots for continuous spline approximation."""
    knots = np.concatenate([[a], theta_start, [b]])

    if track_iterates:
        iterates = [theta_start.copy()]

    # Track the previous d norm and the number of iterations with unchanged norm
    previous_d_norm = None
    unchanged_count = 0

    for iteration in range(100):
        d, S, _, _ = evaluate_and_get_direction(f, knots, m)
        knot_direction = d[-k:]

        d_norm = np.linalg.norm(d)
        if d_norm < 1e-5:
            break

        # Stop if ||d|| has not changed for 5 iterations
        if previous_d_norm is not None and np.isclose(d_norm, previous_d_norm):
            unchanged_count += 1
        else:
            unchanged_count = 0

        if unchanged_count >= 5:
            break

        previous_d_norm = d_norm

        # Use Armijo line search to find the next knots in the direction of d
        knots = armijo_line_search(f, knots, m, knot_direction)

        if track_iterates:
            iterates.append(knots[1:-1].copy())

    # Final evaluation of Psi bar at the last knots
    final_dev, approx, _ = Psi_bar(f, knots, m)

    if track_iterates:
        return knots, S, approx, iteration, final_dev, iterates

    return knots, S, approx, iteration, final_dev


def insert_extra_knots(knots, k):
    """Insert additional knots into the knot vector to ensure there are k internal knots."""
    knots = np.asarray(knots, dtype=float)

    while len(knots) < k + 2:
        # Find the largest gap between consecutive knots
        gaps = np.diff(knots)

        # Find the index of the largest gap
        i = np.argmax(gaps)

        # Insert a new knot in the middle of the largest gap
        new_knot = 0.5 * (knots[i] + knots[i + 1])
        knots = np.insert(knots, i + 1, new_knot)

    return knots


if __name__ == "__main__":
    # Example usage of the descent algorithm with a test function
    function_name = "f_g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]

    a, b = test_functions.INTERVALS[function_name]
    k = 2
    m = 1

    approx, _ = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
    theta_start = approx.g.knots

    # If the number of internal knots is not equal to k, insert extra knots
    if len(theta_start) != k + 2:
        theta_start = insert_extra_knots(theta_start, k)

    # Remove the first and last knots (the endpoints) to get the internal knots
    theta_start = theta_start[1:-1]

    # Run the descent algorithm to find the optimal internal knots
    optimal_knots, S, A, iteration, deviation, iterates = descent_algorithm(
        theta_start, f, a, b, m, k, track_iterates=True
    )

    # Plot the spline approximation
    plotting.plot_single(
        A,
        points=A.basis,
        f_label=function_label,
        approximation_label=rf"$s^*_{{{m}}}(t)$",
        title=f"Descent algorithm for {function_label} with {k} internal knots and degree {m}",
        file_name=f"{function_name}_k{k}_m{m}_test",
    )

    # Load the precomputed Psi bar values for contour plotting
    npz_path = Path(f"psi_surfaces/psi_surface_{function_name}_k{k}_m{m}.npz")
    if npz_path.exists():
        data = np.load(npz_path)
        theta_1_values = data["theta_1_values"]
        theta_2_values = data["theta_2_values"]
        psi_values = data["psi_values"]

    # Plot the contour of Psi bar with the descent path
    plotting.plot_objective_psi_bar_contour(
        theta_1_values,
        theta_2_values,
        psi_values,
        theta_found=optimal_knots[1:-1],
        theta_path=iterates,
        title=r"Descent path on $\overline{\Psi}(\theta)$",
        file_name=f"psi_path_contour_{function_name}_k{k}_m{m}_test",
    )
