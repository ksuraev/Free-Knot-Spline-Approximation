import warnings
from pathlib import Path
from time import perf_counter

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
    if D is not None and U[0] is D:
        return C(U[1:], D)
    return np.concatenate([[x + y for y in C(U[1:], D)] for x in U[0]])


def is_unstable_point(a1, sign):
    return (a1 < 0 and sign > 0) or (a1 > 0 and sign < 0)


# S = {β_j : t ∈ E_smooth ∪ E_neutral ∪ E^+_max ∪ E^-_min}
def build_stable_set_S(basis, S, signs):
    betas = []

    for t, sign in zip(basis, signs):
        knot_indices = np.where(np.isclose(S.knots[1:-1], t))[0]

        if len(knot_indices) > 0:
            k = knot_indices[0] + 1
            t = S.knots[k]

            a1 = S.polynomials[k].coef[1]

            if is_unstable_point(a1, sign):
                continue

            betas.append(beta(S, k - 1, t, sign))
            betas.append(beta(S, k, t, sign))

        else:
            k = np.where(S.knots[:-1] <= t)[0][-1]
            betas.append(beta(S, k, t, sign))

    return betas


# U ={Δ_j : t ∈ E^+_min ∪ E^-_max}
def build_unstable_set_U(basis, S, signs):
    deltas = []

    for t, sign in zip(basis, signs):
        knot_indices = np.where(np.isclose(S.knots[1:-1], t))[0]

        if len(knot_indices) == 0:
            continue

        k = knot_indices[0] + 1
        t = S.knots[k]

        a1 = S.polynomials[k].coef[1]

        if is_unstable_point(a1, sign):
            deltas.append(Delta(S, k, t, sign))

    return deltas


# ∑_{Δ_j ∈ U} Δ_j
def superdifferential(U):
    return C(U, D=None)


def subdifferential(S, U):
    if len(U) == 0:
        return np.array(S)  # no clue if this is right but fails otherwise

    superdiff = superdifferential(U)

    # S - superdifferential
    S_minus_superdiff = [x - y for x in S for y in superdiff]

    # C_Δ = 0 - ∑_{Δ_j ≠ Δ} Δ_j
    zero = np.zeros_like(U[0][0])
    C_delta_union = [
        zero - v for D in U for v in C(U, D)
    ]  # JULIEN!!!!!!!!! check empty U

    return np.concatenate([S_minus_superdiff, C_delta_union])


def closest_subgradient(subdiff, v):
    G = subdiff.T

    P = G.T @ G
    q = -G.T @ v

    A = np.ones((1, G.shape[1]))
    b = np.ones(1)
    lb = np.zeros(G.shape[1])

    problem = qpsolvers.Problem(P, q, None, None, A, b, lb=lb)
    sol = qpsolvers.solve_problem(problem, solver="proxqp", max_iter=20000)
    if sol is None:
        raise RuntimeError("QP returned no solution")
    if not sol.found:
        info = sol.extras["info"]
        raise RuntimeError(
            f"QP failed to find a solution. Status: {info.status}, Iterations: {info.iter}"
        )
    return G @ sol.x


def descent_direction(basis, S, signs):
    S_set = build_stable_set_S(basis, S, signs)
    U_set = build_unstable_set_U(basis, S, signs)

    subdiff = subdifferential(S_set, U_set)
    if len(U_set) == 0:
        superdiff = np.zeros((1, subdiff.shape[1]))
    else:
        superdiff = superdifferential(U_set)

    max_norm = -np.inf
    best_direction = None

    for i, w in enumerate(superdiff):
        v = -w
        u = closest_subgradient(subdiff, v)
        direction = v - u

        if np.linalg.norm(direction) > max_norm:
            max_norm = np.linalg.norm(direction)
            best_direction = direction
        # if (i + 1) % 10 == 0:
        #     print(f"Completed {i + 1}/{len(superdiff)} QPs")

    return best_direction


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
    solver = pl.HiGHS(
        msg=False,
        solver="simplex",
        # primal_feasibility_tolerance=1e-10,
        # dual_feasibility_tolerance=1e-10,
        # optimality_tolerance=1e-10,
    )

    prob.solve(solver)

    status = pl.LpStatus[prob.status]
    if status != "Optimal":
        print(f"LP status: {pl.LpStatus[prob.status]}, knots: {knots}")
        return None, None, None

    # The active constraints should give us the maximum deviation points:
    active = [
        int(name) for name, c in list(prob.constraints.items()) if abs(c.slack) == 0
    ]
    t_active = np.array(
        [
            [samples[i], 1] if i < nsamples else [samples[i - nsamples], -1]
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
    S = Spline.SUSpline.from_coefficients(knots, m, coefs)
    A = Spline.Approximation(f, S, [knots[0], knots[-1]], basis=basis)

    return z.varValue, A, signs


def Psi_bar_dual(f, knots, m, nsamples):
    samples = np.linspace(knots[0], knots[-1], nsamples)
    samples = np.unique(np.concatenate([samples, np.asarray(knots)]))
    nsamples = len(samples)
    A, b = simplex_system(f, samples, knots, m)

    prob = pl.LpProblem("dual_simplex", pl.LpMaximize)

    # one dual variable for each constraint (2N)
    y_plus = pl.LpVariable.dicts("y_p%s", range(nsamples), lowBound=0)
    y_minus = pl.LpVariable.dicts("y_m%s", range(nsamples), lowBound=0)

    # dual objective maximise f(t)(y+ - y-) = b^T y
    prob += pl.lpSum(b[i] * (y_minus[i] - y_plus[i]) for i in range(nsamples))

    # dual constraints
    # M^T(y+ - y-) = 0
    for j in range(A.shape[1]):
        prob += (
            pl.lpSum(A[i, j] * (y_minus[i] - y_plus[i]) for i in range(nsamples)) == 0
        )

    # (y+ + y-) = 1
    prob += pl.lpSum(y_plus[i] + y_minus[i] for i in range(nsamples)) == 1

    solver = pl.HiGHS(msg=False, solver="simplex")
    prob.solve(solver)

    i_plus = [(samples[i], 1) for i in range(nsamples) if y_plus[i].varValue != 0]
    i_minus = [(samples[i], -1) for i in range(nsamples) if y_minus[i].varValue != 0]

    alt_pts = np.concatenate([i_plus, i_minus])
    alt_pts = alt_pts[np.argsort(alt_pts[:, 0])]
    basis = alt_pts[:, 0]  # [t for [t, _] in t_active]
    signs = alt_pts[:, 1]  # [s for [_, s] in t_active]

    # Extract the solution for the coefficients and the deviation
    r = np.array([c.pi for c in list(prob.constraints.values())[:-1]])
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
    d_full = np.concatenate([[0], d, [0]])
    bounds = [
        (knots[i] - knots[i - 1]) / (d_full[i - 1] - d_full[i])
        for i in range(1, len(knots))
        if d_full[i - 1] - d_full[i] > 0
    ]
    alpha = max(bounds) if bounds else 100

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

    for iteration in range(100):
        d, S, _, z = evaluate_and_get_direction(f, knots, m, nsamples=nsamples)
        knot_direction = d[-k:]

        d_norm = np.linalg.norm(knot_direction)
        # if iteration % 10 == 0:
        #     print("d_norm", d_norm)

        if d_norm < 1e-5:
            break

        # Use Armijo line search to find the next knots in the direction of d
        new_knots = armijo_line_search(f, knots, m, knot_direction, nsamples=nsamples)

        if np.allclose(new_knots, knots):
            break
        knots = new_knots

        if track_iterates:
            iterates.append(knots[1:-1].copy())

    # Final evaluation of Psi bar at the last knots
    final_dev, approx, signs = Psi_bar(f, knots, m, nsamples=nsamples)
    S = approx.g

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


def minimal_chain_bisection(f, a, b, m, nsamples=1000):
    theta_min, theta_max = a, b
    eps = 1e-5

    while theta_max - theta_min > eps:
        theta_mid = (theta_min + theta_max) / 2
        knots = [a, theta_mid, b]

        deviation, A, signs = Psi_bar(f, knots, m, nsamples=nsamples)

        counts = A.alternance_per_interval(A.basis, signs)

        if counts[0] >= m + 2 and counts[1] >= m + 2:
            return theta_mid
        if counts[0] >= m + 2:
            theta_max = theta_mid
        elif counts[1] >= m + 2:
            theta_min = theta_mid
        elif counts == [m + 1, m + 1]:
            return theta_mid
        else:
            raise RuntimeError(f"Unexpected alternance counts {counts}")

    return (theta_min + theta_max) / 2


# def redistribute(knots, theta_min, theta_max, i, k, b):
#     knots[i + 1 : k + 1] = np.linspace(knots[i], b, k - i + 2)[1:-1]

#     for j in range(i + 1, k + 1):
#         theta_min[j] = knots[i]
#         theta_max[j] = b


# Increase α and that'll push the knots to the left, decrease it (still >0) and that'll push them to the right.
def redistribute(knots, theta_min, theta_max, i, k, b, alpha=0.6):
    a = knots[i]

    for j in range(i + 1, k + 1):
        knots[j] = a + (b - a) / (k + 2 - j) ** alpha

        theta_min[j] = a
        theta_max[j] = b


def is_minimal_chain(counts, m):
    return (
        len(counts) >= 2
        and counts[0] >= m + 1
        and counts[-1] >= m + 1
        and all(c >= m for c in counts[1:-1])
    )


def minimal_chain_bisection_general(knots, f, a, b, m, k, nsamples=1000):
    eps = 1e-5
    # equidistant knots as initial guess
    # knots = np.linspace(a, b, k + 2)

    theta_min = knots.copy()
    for i in range(1, k + 1):
        theta_min[i] = knots[i - 1]
    theta_max = [b for _ in range(k + 2)]

    # Find minimal chain
    for iter in range(1000):
        if np.any(np.diff(knots) <= 0):
            raise ValueError(f"Knots are not strictly increasing: {knots}")
        old_knots = knots.copy()
        old_min = np.asarray(theta_min).copy()
        old_max = np.asarray(theta_max).copy()

        z, A, signs = Psi_bar_dual(f, knots, m, nsamples=nsamples)
        counts = A.alternance_per_interval(A.basis, signs)

        # print(f"\nknots={knots}, counts={counts}")
        # print("theta_min:", np.asarray(theta_min))
        # print("theta_max:", np.asarray(theta_max))

        if is_minimal_chain(counts, m):
            print("minimal chain found. Count:", counts)
            print(f"iterations: {iter}")
            return knots[1:-1]

        # select the case/move first
        move = None

        # Case 1: [theta_k, b] contains m+2
        if counts[-1] >= m + 2:
            # move the last knot to the right
            move = ("right", k)
        # Case 2: [theta_i-1, theta_i] contains m+2
        if move is None:
            for i in range(1, k + 1):
                if counts[i - 1] >= m + 2:
                    move = ("left", i)
                    break
        # Case 3: [theta_i, b] contains minimal chain
        if move is None:
            for i in range(k - 1, 0, -1):
                if is_minimal_chain(counts[i:], m):
                    move = ("right", i)
                    break
        # Case 4: [theta_i, theta_j] contains minimal chain
        if move is None:
            for j in range(k, 1, -1):
                for i in range(j - 2, -1, -1):
                    if is_minimal_chain(counts[i:j], m):
                        move = ("left", j)
                        break
                if move is not None:
                    break

        updated = False
        converged = None

        if move is not None:
            direction, i = move

            if direction == "left":
                lb = max(
                    theta_min[i], knots[i - 1]
                )  # issue with strict ordering of knots sometimes
                # lb = theta_min[i]
                ub = knots[i]
            else:
                lb = knots[i]
                ub = theta_max[i]

            if ub - lb > eps:
                if direction == "left":
                    theta_max[i] = knots[i]
                else:
                    theta_min[i] = knots[i]

                knots[i] = (lb + ub) / 2

                # redistribute the later knots to maintain ordering
                if i < k:
                    redistribute(knots, theta_min, theta_max, i, k, b)
                updated = True
            else:
                converged = i

        if converged is not None and not updated:
            # update the previous knot if the current knot has converged
            for j in range(converged - 1, 0, -1):
                lb = knots[j]
                ub = theta_max[j]

                if ub - lb <= eps:
                    continue

                theta_min[j] = knots[j]
                knots[j] = (lb + ub) / 2

                redistribute(knots, theta_min, theta_max, j, k, b)

                updated = True
                break
        if not updated:
            print(f"No further update possible. Counts: {counts}")
            print(f"iterations: {iter}")
            return knots[1:-1]
        if (
            np.array_equal(knots, old_knots)
            and np.array_equal(theta_min, old_min)
            and np.array_equal(theta_max, old_max)
        ):
            print(f"Bisection stalled. Counts: {counts}")
            print(f"Knots: {knots}")
            print(f"iterations: {iter}")
            return knots[1:-1]

    print(f"Maximum iterations reached. Counts: {counts}")
    print(f"iterations: {iter}")
    return knots[1:-1]


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]

    a, b = test_functions.INTERVALS[function_name]
    k = 10
    m = 1

    x_min_knots, x_max_knots = nurnberger_mod.get_knots(f, a, b, k, m)

    if len(x_min_knots) != k:
        x_min_knots = [a, *x_min_knots, b]
        print(
            f"Number of knots ({len(x_min_knots)}) does not match expected number ({k + 2}). Inserting extra knots."
        )
        x_min_knots = insert_extra_knots(x_min_knots, k)
    else:
        x_min_knots = [a, *x_min_knots, b]

    # print("x_min_knots", np.asarray(x_min_knots))

    # min_chain_knots = minimal_chain_bisection_general(
    #     x_min_knots, f, a, b, m, k, nsamples=1000
    # )
    # print(f"Initial knots: {min_chain_knots}")

    # min_chain_knots = [a, *min_chain_knots, b]

    # initial_z, initial_A, initial_signs = Psi_bar(f, min_chain_knots, m, nsamples=1000)
    # print("initial z", initial_z)
    # print(
    #     "initial counts",
    #     initial_A.alternance_per_interval(initial_A.basis, initial_signs),
    # )

    opt_knots, signs, S, A, iterations, z = descent_algorithm(
        x_min_knots[1:-1], f, a, b, m, k, nsamples=1000, track_iterates=False
    )

    print(f"Final knots: {opt_knots}")
    print(f"Final z={z:.12f}, counts={A.alternance_per_interval(A.basis, signs)}")

    # x_min_knots, x_max_knots = nurnberger_mod.get_knots(f, a, b, k, m)

    # if len(x_min_knots) != k:
    #     x_min_knots = [a, *x_min_knots, b]
    #     print(
    #         f"Number of knots ({len(x_min_knots)}) does not match expected number ({k + 2}). Inserting extra knots."
    #     )
    #     x_min_knots = insert_extra_knots(x_min_knots, k)
    # else:
    #     x_min_knots = [a, *x_min_knots, b]

    # print("x_min_knots", np.asarray(x_min_knots))

    # initial_x_min_z, initial_x_min_A, initial_x_min_signs = Psi_bar(
    #     f, x_min_knots, m, nsamples=1000
    # )
    # initial_x_min_counts = initial_x_min_A.alternance_per_interval(
    #     initial_x_min_A.basis, initial_x_min_signs
    # )
    # print("x_min counts", initial_x_min_counts)
    # print("x min initial z", initial_x_min_z)

    # plotting.plot_single(
    #     initial_x_min_A,
    #     points=initial_x_min_A.basis,
    #     file_name=f"z_{function_name}_k{k}_m{m}_initial_cvxopt",
    #     title=f"Initial. z={initial_x_min_z:.4f}, counts={initial_x_min_counts}",
    # )

    # opt_knots, signs, S, A, iterations, z = descent_algorithm(
    #     x_min_knots[1:-1], f, a, b, m, k, nsamples=1000, track_iterates=False
    # )
    # print(f"Final knots: {opt_knots}")
    # print(f"Final z={z:.12f}, counts={A.alternance_per_interval(A.basis, signs)}")

    # plotting.plot_single(
    #     A,
    #     points=A.basis,
    #     file_name=f"z_{function_name}_k{k}_m{m}_final_cvxopt",
    #     title=f"Final. z={z:.4f}, counts={A.alternance_per_interval(A.basis, signs)}",
    # )
