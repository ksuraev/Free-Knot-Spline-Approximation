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


# β_j (𝜃,a,t)= sign * ∇_{θ,a} P_j(t)
def beta(S, j, t, sign):
    m = S.degree

    wrt_coeffs = [
        np.maximum(0, t - knot) ** d for knot in S.knots[:-1] for d in range(1, m + 1)
    ]

    wrt_knots = [-p.deriv()(t) for p in S.polynomials[1 : j + 1]]
    wrt_knots = np.concatenate((wrt_knots, np.zeros(S.nintervals() - j - 1)))

    return sign * np.concatenate(([1], wrt_coeffs, wrt_knots))


# Δ_j = co{β_j, β_j+1}
def Delta(S, j, t, sign):
    return [beta(S, j - 1, t, sign), beta(S, j, t, sign)]


def C(U, D=None):
    if len(U) == 0:
        return [0]  # single element? or array of zeros?
    if D is not None and U[0] == D:
        return C(U[1:], D)
    return np.concatenate([[x + y for y in C(U[1:], D)] for x in U[0]])


def is_unstable_point(a1, sign):
    return (a1 < 0 and sign > 0) or (a1 > 0 and sign < 0)


# S = {β_j : t ∈ E_smooth ∪ E_neutral ∪ E^+_max ∪ E^-_min}
def build_stable_set_S(basis, S, signs):
    betas = []

    for t, sign in zip(basis, signs):
        k = np.where(S.knots[:-1] <= t)[0][
            -1
        ]  # from Spline.py. t close to internal knot wont return the right k?

        if 0 < k < S.nintervals() and np.isclose(t, S.knots[k]):
            a1 = S.polynomials[k].coef[1]

            if is_unstable_point(a1, sign):
                continue
            betas.append(beta(S, k - 1, t, sign))

        betas.append(beta(S, k, t, sign))

    return betas


# U ={Δ_j : t ∈ E^+_min ∪ E^-_max}
def build_unstable_set_U(basis, S, signs):
    deltas = []

    for t, sign in zip(basis, signs):
        k = np.where(S.knots[:-1] <= t)[0][-1]

        if not 0 < k < S.nintervals() or not np.isclose(t, S.knots[k]):
            continue

        a1 = S.polynomials[k].coef[1]

        if is_unstable_point(a1, sign):
            deltas.append(Delta(S, k, t, sign))

    return deltas


def subdifferential(S, U):
    # ∑_{Δ_j ∈ U} Δ_j
    superdiff = C(U, D=None)

    # S - superdifferential
    S_minus_superdiff = [x - y for x in S for y in superdiff]

    # C_Δ = 0 - ∑_{Δ_j ≠ Δ} Δ_j
    C_delta_union = [v for D in U for v in -C(U, D)]  # JULIEN!!!!!!!!! check empty U

    return np.concatenate([S_minus_superdiff, C_delta_union])


def subgradients(basis, S, signs):
    """Compute the subgradients"""
    P = GRA.build_P(basis, S.knots, S.degree)
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
    S_set = build_stable_set_S(basis, S, signs)
    U = build_unstable_set_U(basis, S, signs)
    print("Stable points:", len(S_set))
    print("Unstable points:", len(U))

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
    samples = np.linspace(knots[0], knots[-1], nsamples)

    # Include knots in the samples
    samples = np.unique(np.concatenate([samples, np.asarray(knots)]))
    nsamples = len(samples)

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

    # print("Active points (basis and signs):")
    # for t, sign in zip(basis, signs):
    #     print(f"t={t:+.8f}, sign={sign:+.0f} ")

    # Extract the coefficients from the solution and construct the spline approximation
    coefs = np.array([x[j].varValue for j in range(A.shape[1])])
    # print(f"Coefficients: {coefs}")
    S = Spline.SUSpline.from_coefficients(knots, m, coefs)
    A = Spline.Approximation(f, S, [knots[0], knots[-1]], basis=basis)

    # sample_step = samples[1] - samples[0]
    # bdev = abs(A.deviation(basis))
    # tol = abs(sample_step) * 1e-6
    # bsel0 = basis[1] - basis[0] > sample_step + tol or bdev[0] > bdev[1]
    # bsel1 = (bdev[1:-1] > bdev[:-2]) | (basis[1:-1] - basis[:-2] > sample_step + tol)
    # bsel2 = (bdev[1:-1] > bdev[2:]) | (basis[2:] - basis[1:-1] > sample_step + tol)
    # bselend = basis[-1] - basis[-2] > sample_step + tol or bdev[-2] > bdev[-2]
    # bsel = np.concatenate(([bsel0], bsel1 & bsel2, [bselend]))

    # new_basis = basis[bsel]
    # new_signs = signs[bsel]

    # sample_step = samples[1] - samples[0]
    # new_basis = []
    # new_signs = []
    # i = j = 0
    # while i < len(basis):
    #     while (
    #         j + 1 < len(basis)
    #         and np.isclose(basis[j + 1] - basis[j], sample_step)
    #         and abs(A.deviation(basis[j + 1])) > abs(A.deviation(basis[j]))
    #     ):
    #         j = j + 1
    #     j_max = j
    #     while (
    #         j + 1 < len(basis)
    #         and np.isclose(basis[j + 1] - basis[j], sample_step)
    #         and abs(A.deviation(basis[j + 1])) < abs(A.deviation(basis[j]))
    #     ):
    #         j = j + 1

    #     new_basis.append(basis[j_max])
    #     new_signs.append(signs[j_max])
    #     i = j = j + 1
    # A.basis = new_basis

    return z.varValue, A, signs


def Psi_bar_dual(f, knots, m, nsamples):
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

        d_norm = np.linalg.norm(knot_direction)

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

    if track_iterates:
        return knots, signs, S, approx, iteration, final_dev, iterates

    return knots, signs, S, approx, iteration, final_dev


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


def minimal_chain_bisection(f, function_label, a, b, k, m):
    theta_min, theta_max = a, b
    eps = 1e-5
    iteration = 0
    nsamples = 10000

    def result_dict(
        theta_mid,
        exit_reason,
        counts,
        deviation,
        A,
        signs,
        optimal_knots=None,
        A_after=None,
        signs_after=None,
        descent_iterations=None,
        final_deviation=None,
    ):
        return {
            "theta_min": theta_min,
            "theta_max": theta_max,
            "theta_mid": theta_mid,
            "bisection_iterations": iteration,
            "exit_reason": exit_reason,
            "counts_before_descent": counts,
            "deviation_before_descent": deviation,
            "alt_pts_before_descent": A.basis if A is not None else None,
            "signs_before_descent": signs,
            "optimal_knots": optimal_knots,
            "counts_after_descent": (
                A_after.alternance_per_interval(A_after.basis, signs_after)
                if A_after is not None and signs_after is not None
                else None
            ),
            "descent_iterations": descent_iterations,
            "deviation_after_descent": final_deviation,
            "alt_pts_after_descent": (A_after.basis if A_after is not None else None),
        }

    while theta_max - theta_min > eps:
        theta_mid = (theta_min + theta_max) / 2
        knots = [a, theta_mid, b]

        deviation, A, signs = Psi_bar(f, knots, m, nsamples=nsamples)

        counts = A.alternance_per_interval(A.basis, signs)

        if counts[0] >= m + 2 and counts[1] >= m + 2:
            plotting.plot_single(
                A,
                points=A.basis,
                file_name=f"{function_label}_minimal_chain_bisection_k{k}_m{m}",
            )
            return result_dict(theta_mid, "m+2|m+2", counts, deviation, A, signs)
        if counts[0] >= m + 2:
            theta_max = theta_mid
        elif counts[1] >= m + 2:
            theta_min = theta_mid
        elif counts == [m + 1, m + 1]:
            optimal_knots, signs_after, S, A_after, descent_iterations, final_z = (
                descent_algorithm([theta_mid], f, a, b, m, k, nsamples=nsamples)
            )
            z, A_after, signs_after = Psi_bar(f, optimal_knots, m, nsamples=nsamples)
            plotting.plot_single(
                A_after,
                points=A_after.basis,
                file_name=f"{function_label}_minimal_chain_bisection_k{k}_m{m}",
            )
            return result_dict(
                theta_mid,
                "m+1|m+1",
                counts,
                deviation,
                A,
                signs,
                optimal_knots=optimal_knots,
                A_after=A_after,
                signs_after=signs_after,
                descent_iterations=descent_iterations,
                final_deviation=final_z,
            )
        else:
            return result_dict(
                theta_mid, "something weird", counts, deviation, A, signs
            )
        iteration += 1

    theta_mid = (theta_min + theta_max) / 2
    deviation, A, signs = Psi_bar(f, [a, theta_mid, b], m, nsamples=nsamples)
    plotting.plot_single(
        A,
        points=A.basis,
        file_name=f"{function_label}_minimal_chain_bisection_k{k}_m{m}",
    )
    counts = A.alternance_per_interval(A.basis, signs)
    if counts[0] >= m + 2 and counts[1] >= m + 2:
        return result_dict(theta_mid, "m+2|m+2", counts, deviation, A, signs)
    else:
        return result_dict(
            theta_mid, "theta_max - theta_min < eps", counts, deviation, A, signs
        )


if __name__ == "__main__":
    function_name = "cos"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]

    a, b = test_functions.INTERVALS[function_name]
    k = 1
    m = 1

    # knots = [a, 3, b]
    # z, A, signs = Psi_bar(f, knots, m, nsamples=1000)
    # knots = [0.2855]
    # result = minimal_chain_bisection(f, function_name, a, b, k, m)
    # knots = [a, result["theta_mid"], b]
    # z, A, signs = Psi_bar(f, knots, m, nsamples=1000)

    x_min_knots, x_max_knots = nurnberger_mod.get_knots(f, a, b, k, m)
    print(f"x_min_knots={x_min_knots}, x_max_knots={x_max_knots}")

    if len(x_min_knots) != k:
        x_min_knots = insert_extra_knots(x_min_knots, k)

    opt_knots, signs, S, A, iterations, z, iterates = descent_algorithm(
        x_min_knots, f, a, b, m, k, nsamples=1001, track_iterates=True
    )
    print(f"opt knots={opt_knots}")
    print(f"alt pts={A.basis}")
    print(f"z={z}, counts={A.alternance_per_interval(A.basis, signs)}")

    # plotting.plot_single(
    #     A,
    #     points=A.basis,
    #     file_name=f"z_{function_name}_k{k}_m{m}_1000_1e-4_3-2",
    # )

    npz_path = Path(
        f"psi_surfaces_10000_simplex/psi_surface_{function_name}_k{k}_m{m}_10000.npz"
    )
    data = np.load(npz_path)
    thetas = data["theta_values"]
    psi_values = data["psi_values"]

    plotting.plot_objective_psi_bar(
        thetas,
        psi_values,
        opt_knots[1],
        z,
        theta_path=iterates,
        file_name=f"psi_{function_name}_k{k}_m{m}_descent_path",
    )

    # plotting.plot_psi_bar_regions(
    #     npz, file_name=f"z_psi_bar_{function_name}_k{k}_m{m}_1000_1e-4"
    # )
    # result = minimal_chain_bisection(f, function_name, a, b, k, m)
    # for key, value in result.items():
    #     print(f"{key}: {value}")
