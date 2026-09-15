# extensions - subgradients and simplex system
import numpy as np
from qpsolvers import solve_qp

import nadia_original
import test_functions


# Where is signs coming from
def build_gradients(basis, S, signs):
    P = nadia_original.build_P(basis, S.knots, S.degree)
    P = np.transpose(P)

    M = np.zeros((len(S.knots) - 2, P.shape[1]))
    a = S.polynomials  # ?
    for i, knot in enumerate(S.knots[1:-1]):
        col = 0
        for b in basis:
            for t in b:
                if t > knot:
                    M[i, col] = sum(
                        -(j + 1) * a[i, j] * (t - knot) ** j for j in range(m)
                    )

                col += 1

    G = np.concatenate([np.ones((1, P.shape[1])), P, M], axis=0)
    G = np.multiply(G, signs)

    return G


def find_descent_direction():
    pass


def build_simplex_system(f, samples, knots, m):
    P = nadia_original.build_P(samples, knots, m)
    M = np.concatenate([np.ones((len(samples), 1)), P], axis=1)

    # Stack M and -M vertically
    A = np.concatenate([M, -M], axis=0)

    # add column of ones
    A = np.concatenate([A, -np.ones((2 * len(samples), 1))], axis=1)

    b = np.concatenate([f(samples), -f(samples)], axis=0)
    return A, b


if __name__ == "__main__":
    # Example usage
    function_name = "sin"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 1
    m = 1

    samples = np.linspace(a, b, 100)
    knots = np.linspace(a, b, k + 2)

    A, b = build_simplex_system(f, samples, knots, m)
    print("A:\n", A.shape)
    print("b:\n", b.shape)
