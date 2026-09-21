# extensions - subgradients and simplex system
import numpy as np
import pulp as pl
from qpsolvers import solve_qp

import nadia_original
import nurnberger_mod
import plotting
import Spline
import test_functions


def build_gradients(basis, S, signs):
    P = nadia_original.build_P(basis, S.knots, S.degree)
    P = P.T

    M = np.zeros((len(S.knots) - 2, P.shape[1]))
    a = [p.coef[1:] for p in S.polynomials]
    for i, knot in enumerate(S.knots[1:-1]):
        col = 0
        for t in basis:
            if t > knot:
                M[i, col] = sum(-(j + 1) * a[i][j] * (t - knot) ** j for j in range(m))

            col += 1

    G = np.concatenate([np.ones((1, P.shape[1])), P, M], axis=0)
    G = np.multiply(G, signs)

    return G


def find_descent_direction(basis, S, signs):
    G = build_gradients(basis, S, signs)
    P = G.T @ G
    q = np.zeros(G.shape[1])
    A = np.ones(G.shape[1])
    b = np.ones(1)
    #x = solve_qp(P, q, None, None, A, b, lb=q, solver="cvxopt")
    x = solve_qp(P, q, None, None, A, b, lb=q, solver="clarabel")
    v = G@x
    #print(f"Vector: {G.T @ v}, norm: {v@v}")

    return -v


# [-len(S.knots) + 2 :]


def build_simplex_system(f, samples, knots, m):
    P = nadia_original.build_P(samples, knots, m)
    M = np.concatenate([np.ones((len(samples), 1)), P], axis=1)

    # Stack M and -M vertically
    A = np.concatenate([M, -M], axis=0)

    # Stack f(samples) and -f(samples) vertically to create b
    b = np.concatenate([f(samples), -f(samples)], axis=0)

    return A, b


def solve_simplex(f, knots, m):
    nsamples = 1000
    samples = np.linspace(knots[0], knots[-1], nsamples)
    A, b = build_simplex_system(f, samples, knots, m)

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
        constraint = pl.lpSum(A[i, j] * x[j] for j in range(A.shape[1])) - z <= b[i], f"{i}"
        prob += constraint

    solver = pl.HiGHS(msg=False)
    prob.solve(solver)

    # The active constraints should give us the maximum deviation points:
    active = [int(name) for name, c in list(prob.constraints.items()) if c.slack == 0]
    t_active = np.array([[samples[i], -1] if i<nsamples else [samples[i - nsamples], 1] for i in active])
    t_active = t_active[np.argsort(t_active[:,0])]
    basis = t_active[:,0] # [t for [t, _] in t_active]
    signs = t_active[:,1] # [s for [_, s] in t_active]

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


def directional_derivative(f, knots, m, d_full, h=0.0001):
    psi_theta, _, _ = solve_simplex(f, knots, m)[1]
    psi_new, _, _ = solve_simplex(f, knots + h * d_full, m)[1]

    return (psi_new - psi_theta) / (h * np.linalg.norm(d_full))


def armijo(f, S, knots, m, d, rho=0.5, c=0.1, verbose=False):
    """Armijo line search to find the next theta in the direction of d."""
    # Initialise the step size
    alpha = 10.0
    d_full = np.concatenate([[0], d, [0]])
    directional_deriv = directional_derivative(f, knots, m, d_full)
    print(f"Directional derivative: {directional_deriv:.5f}")
    curr_knots = knots.copy()
    psi_theta, _, _ = solve_simplex(f, curr_knots, m)[1]

    while alpha > 1e-8:
        # Compute the next knots using the step size alpha
        next_knots = curr_knots + alpha * d_full

        # Check if the new knots are valid
        if np.any(next_knots[:-1] >= next_knots[1:]):
            alpha *= rho
            continue

        psi_next, _, _ = solve_simplex(f, next_knots, m)[1]
        # print(
        #     f"alpha: {alpha:.5f}, psi_next: {psi_next:.5f}, psi_theta: {psi_theta:.5f}"
        # )

        # Check the Armijo condition
        if psi_next <= psi_theta + c * alpha * directional_deriv:
            return next_knots

        # If the Armijo condition is not satisfied, reduce alpha and try again
        alpha *= rho

    return curr_knots


def get_direction(f, knots, m):
    # eval Psi
    deviation, approx, signs = solve_simplex(f, knots, m)

    S = approx.g
    # all_pts = approx.maxdeviationpoints()
    basis = approx.basis

    d = find_descent_direction(basis, S, signs)
    return d, S


def descent_algo(x_min, f, a, b, m, k):
    knots = np.linspace(a, x_min, k + 1)
    knots = np.concatenate([knots, [b]])

    for iteration in range(10):
        d, S = get_direction(f, knots, m)
        # directional_deriv = directional_derivative(f, knots, m, d)
        # print(f"{iteration}: directional derivative = {directional_deriv:.5f}")
        print(f"{iteration}: norm d = {np.linalg.norm(d):.5f}")
        if np.linalg.norm(d) < 1e-5:
            break
        knots = armijo(f, S, knots, m, d)

    return knots, S, iteration


if __name__ == "__main__":
    function_name = "f_g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 2
    m = 2

    approx, x_min = nurnberger_mod.discontinuous_spline(f, a, b, k, m)
    # print(f"x_min: {x_min}")
    # if x_min is None:
    #     x_min = approx.g.knots[1]

    # new_knots, S, iteration = descent_algo(x_min, f, a, b, m, k)
    knots = np.linspace(a, x_min, k + 1)
    knots = np.concatenate([knots, [b]])

    z, approx, signs = solve_simplex(f, knots, m)
    S = approx.g
    plotting.plot_duo(
        approx,
        title=f"Descent algorithm for {f_label} with {k} internal knots and degree {m}",
        file_name=f"descent_{function_name}_k{k}_m{m}",
    )

    maxdev_before = approx.maxdeviation()
    #print(f"Max deviation before descent: {maxdev_before[2]:.5f}")
    basis = approx.basis
    g = find_descent_direction(basis, S, signs)
    print(f"Descent direction: {g}")
    h = 1.0
    new_knots = knots + h * np.concatenate([[0], g[-len(knots) + 2 :], [0]])

    a = np.concatenate([p.coef[1:] for p in S.polynomials])
    new_a = a + h * g[1 : -len(knots) + 2]
    newnew_a = new_a.reshape(len(knots) - 1, m)
    # print(f"New coefficients: {newnew_a}")
    newnew_a = [np.concatenate([[0], c]) for c in newnew_a]
    newnew_a[0][0] = S.polynomials[0].coef[0] + h * g[0]
    spline = Spline.SUSpline(
        new_knots,
        [Spline.Polynomial(c, offset=x) for c, x in zip(newnew_a, new_knots[:-1])],
    )
    approx = Spline.Approximation(f, spline, (a, b), basis=None)
    maxdev = approx.maxdeviation()
    print(f"Max deviation: before: {maxdev_before[2]}, and after: {maxdev[2]:.5f}")
    print(f"Improvement (positive is good): {maxdev_before[2] - maxdev[2]:.5f}")
    # print(f"Descent algorithm completed in {iteration} iterations.")
    # approx = Spline.Approximation(f, S, (a, b), basis=None)
    # plotting.plot_duo(
    #     approx,
    #     title=f"Descent algorithm for {f_label} with {k} internal knots and degree {m}",
    #     file_name=f"descent_{function_name}_k{k}_m{m}",
    # )
