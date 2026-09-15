# extensions - subgradients and simplex system
import numpy as np
import pulp as pl
from qpsolvers import solve_qp

import nadia_original
import Spline
import test_functions


def build_gradients(basis, S, signs):
    P = nadia_original.build_P(basis, S.knots, S.degree)
    P = np.transpose(P)

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
    P = G.transpose() @ G
    q = np.zeros(G.shape[1])
    A = np.ones(G.shape[1])
    b = np.ones(1)
    x = solve_qp(P, q, None, None, A, b, lb=q, solver="cvxopt")

    return (-G @ x)[-len(S.knots) + 2 :]


def build_simplex_system(f, samples, knots, m):
    P = nadia_original.build_P(samples, knots, m)
    M = np.concatenate([np.ones((len(samples), 1)), P], axis=1)

    # Stack M and -M vertically
    A = np.concatenate([M, -M], axis=0)

    # Stack f(samples) and -f(samples) vertically to create b
    b = np.concatenate([f(samples), -f(samples)], axis=0)

    return A, b


def solve_simplex(f, samples, knots, m):
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
        constraint = pl.lpSum(A[i, j] * x[j] for j in range(A.shape[1])) - z <= b[i]
        prob += constraint

    solver = pl.HiGHS_CMD(msg=True)
    prob.solve(solver)

    # Extract the solution for the coefficients
    r = np.array([x[j].varValue for j in range(A.shape[1])])
    a = np.concatenate([r[1:].reshape(len(knots) - 1, m)])
    coeffs = [np.concatenate([[0], c]) for c in a]
    coeffs[0][0] = r[0]

    # Create the spline from the coefficients and knots
    polynomials = [Spline.Polynomial(c, offset=x) for c, x in zip(coeffs, knots[:-1])]
    S = Spline.SUSpline(knots, polynomials)

    return S, z.varValue


if __name__ == "__main__":
    function_name = "f_g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 5
    m = 2

    samples = np.linspace(a, b, 1000)
    knots = np.linspace(a, b, k + 2)

    S, deviation = solve_simplex(f, samples, knots, m)
    approx = Spline.Approximation(f, S, (a, b), basis=None)
    all_pts = approx.maxdeviationpoints()
    basis = [t for [t, d] in all_pts]
    signs = [np.sign(d) for [t, d] in all_pts]

    d = find_descent_direction(basis, S, signs)
    print("Descent direction:", d)
