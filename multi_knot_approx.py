import warnings
from pathlib import Path

import numpy as np
import pulp as pl
import qpsolvers
from qpsolvers.conversions.ensure_sparse_matrices import SparseConversionWarning

import GRA
import nurnberger_mod
import plotting
import Spline
import test_functions

warnings.filterwarnings("ignore", category=SparseConversionWarning)


def subgradients(basis, S, signs):
    """Compute the subgradients"""
    P = GRA.build_P(basis, S.knots, S.degree)
    P = P.T
    print(len(S.knots))
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
    P = GRA.build_P(samples, knots, m)
    M = np.concatenate([np.ones((len(samples), 1)), P], axis=1)

    # Stack M and -M vertically
    A = np.concatenate([M, -M], axis=0)

    # Stack f(samples) and -f(samples) vertically to create b
    b = np.concatenate([f(samples), -f(samples)], axis=0)

    return A, b


def Psi_bar(f, knots, m, nsamples):
    """Evaluate Psi bar at the given knots and return the approximation and signs of the deviation."""
    # nsamples = 1000
    samples = np.linspace(knots[0], knots[-1], nsamples)
    A, b = simplex_system(f, samples, knots, m)

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

    status = pl.LpStatus[prob.status]
    if status != "Optimal":
        return None, None, None

    # The active constraints should give us the maximum deviation points:
    active = [int(name) for name, c in list(prob.constraints.items()) if c.slack == 0]
    print(active)
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

    # Extract the coefficients from the solution and construct the spline approximation
    coefs = np.array([x[j].varValue for j in range(A.shape[1])])
    print(f"Coefficients: {coefs}")
    S = Spline.SUSpline.from_coefficients(knots, m, coefs)
    A = Spline.Approximation(f, S, [knots[0], knots[-1]], basis=basis)

    return z.varValue, A, signs


def Psi_bar_dual(f, knots, m, nsamples):
    # nsamples = 1000
    samples = np.linspace(knots[0], knots[-1], nsamples)
    A, b = simplex_system(f, samples, knots, m)

    prob = pl.LpProblem("dual_simplex", pl.LpMaximize)

    # one dual variable for each constraint (2N)
    y_plus = pl.LpVariable.dicts("y_p%s", range(nsamples), lowBound=0)
    y_minus = pl.LpVariable.dicts("y_m%s", range(nsamples), lowBound=0)

    # dual objective maximise f(t)(y+ - y-) = b^T y
    prob += pl.lpSum(b[i] * (y_plus[i] - y_minus[i]) for i in range(nsamples))

    # dual constraints
    # M^T(y+ - y-) = 0
    for j in range(A.shape[1]):
        prob += (
            pl.lpSum(A[i, j] * (y_plus[i] - y_minus[i]) for i in range(nsamples)) == 0
        )

    # (y+ + y-) = 1
    prob += pl.lpSum(y_plus[i] + y_minus[i] for i in range(nsamples)) == 1

    solver = pl.HiGHS(msg=False)
    prob.solve(solver)

    i_plus = [(i, 1) for i in range(nsamples) if y_plus[i].varValue > 1e-12]
    i_minus = [(i, -1) for i in range(nsamples) if y_minus[i].varValue > 1e-12]

    alt_pts = np.concatenate([i_plus, i_minus])
    alt_pts = alt_pts[np.argsort(alt_pts[:, 0])]
    basis = alt_pts[:, 0]  # [t for [t, _] in t_active]
    signs = alt_pts[:, 1]  # [s for [_, s] in t_active]

    # Extract the solution for the coefficients and the deviation
    r = [-c.pi for c in prob.constraints()]
    a = np.concatenate([r[1:].reshape(len(knots) - 1, m)])
    coeffs = [np.concatenate([[0], c]) for c in a]
    coeffs[0][0] = r[0]

    # Create the spline from the coefficients and knots
    polynomials = [Spline.Polynomial(c, offset=x) for c, x in zip(coeffs, knots[:-1])]
    S = Spline.SUSpline(knots, polynomials)
    A = Spline.Approximation(f, S, [knots[0], knots[-1]], basis=basis)

    obj = pl.value(prob.objective)

    return obj, A, signs


def armijo_line_search(f, knots, m, d, nsamples, rho=0.5, c=0.1):
    """Armijo line search to find the next theta in the direction of d."""
    # Initialise the step size
    alpha = 100.0
    d_full = np.concatenate([[0], d, [0]])
    curr_knots = knots.copy()
    psi_theta, _, _ = Psi_bar(f, curr_knots, m, nsamples=nsamples)

    while alpha > 1e-8:
        # Compute the next knots using the step size alpha
        next_knots = curr_knots + alpha * d_full

        # Check if the new knots are valid
        if np.any(next_knots[:-1] >= next_knots[1:]):
            alpha *= rho
            continue

        psi_next, _, _ = Psi_bar(f, next_knots, m, nsamples=nsamples)

        # Check the Armijo condition
        if psi_next <= psi_theta + c * alpha * (np.linalg.norm(d) ** 2):
            return next_knots

        # If the Armijo condition is not satisfied, reduce alpha and try again
        alpha *= rho

    return curr_knots


def evaluate_and_get_direction(f, knots, m, nsamples):
    """Evaluate Psi bar at the given knots and return the descent direction, spline, approximation, and deviation."""
    # Evaluate Psi bar at the given knots
    deviation, approx, signs = Psi_bar(f, knots, m, nsamples=nsamples)

    # Compute the descent direction using the subgradients and the simplex system
    S = approx.g
    basis = approx.basis
    d = descent_direction(basis, S, signs)

    return d, S, approx, deviation


def descent_algorithm(theta_start, f, a, b, m, k, nsamples=1000, track_iterates=False):
    """Descent algorithm to find the optimal internal knots for continuous spline approximation."""
    knots = np.concatenate([[a], theta_start, [b]])

    if track_iterates:
        iterates = [theta_start.copy()]

    # Track the previous d norm and the number of iterations with unchanged norm
    previous_d_norm = None
    unchanged_count = 0

    for iteration in range(100):
        d, S, _, _ = evaluate_and_get_direction(f, knots, m, nsamples=nsamples)
        knot_direction = d[-k:]
        print(f"knot direction: {knot_direction}")

        # eps = 1e-3
        # knots_plus = knots.copy()
        # knots_minus = knots.copy()
        # knots_plus[1:-1] += eps * knot_direction
        # knots_minus[1:-1] -= eps * knot_direction
        # z_plus, _, _ = Psi_bar(f, knots_plus, m, nsamples=nsamples)
        # z_minus, _, _ = Psi_bar(f, knots_minus, m, nsamples=nsamples)
        # if z_minus < z_plus:
        #     knot_direction = -knot_direction

        d_norm = np.linalg.norm(knot_direction)
        print(f"Iteration {iteration}: ||d|| = {d_norm:.6f}")
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
        knots = armijo_line_search(f, knots, m, knot_direction, nsamples=nsamples)

        if track_iterates:
            iterates.append(knots[1:-1].copy())

    # Final evaluation of Psi bar at the last knots
    final_dev, approx, signs = Psi_bar(f, knots, m, nsamples=nsamples)
    print(f"signs: {signs}")

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
    function_name = "sin_if_else"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]

    a, b = test_functions.INTERVALS[function_name]
    k = 1
    m = 3

    x_min_knots, x_max_knots = nurnberger_mod.get_knots(f, a, b, k, m)
    theta_start = x_min_knots

    # If the number of internal knots is not equal to k, insert extra knots
    if len(theta_start) != k:
        theta_start = insert_extra_knots(theta_start, k)

    # z, A, signs = Psi_bar(f, np.concatenate([[a], theta_start, [b]]), m)

    zd, Ad, signsd = Psi_bar_dual(f, np.concatenate([[a], theta_start, [b]]), m)

    # # Run the descent algorithm to find the optimal internal knots
    optimal_knots, S, A, iteration, deviation, iterates = descent_algorithm(
        theta_start, f, a, b, m, k, track_iterates=True
    )

    print(f"Optimal internal knots: {optimal_knots[1:-1]}")
    print(f"Optimal Psi bar value: {deviation}")
    print(f"basis/alt pts: {A.basis}")

    # # Plot the spline approximation
    plotting.plot_single(
        A,
        points=A.basis,
        f_label=function_label,
        approximation_label=rf"$s^*_{{{m}}}(t)$",
        title=f"Descent algorithm for {function_label} with {k} internal knots and degree {m}",
        file_name=f"{function_name}_k{k}_m{m}",
    )
