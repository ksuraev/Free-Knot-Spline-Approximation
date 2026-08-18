import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x ** (1 / 2)


a, b = 0, 1
k = 2  # number of free knots (not including a and b)
m = 2  # degree of polynomial to fit in each subinterval (assumed constant throughout)

# Choose intial knots as equidistant points
knots = np.linspace(a, b, k + 2)

n = len(knots) - 1  # number of subintervals

# Form intial basis - m per internal subinterval, m+1 per endpoint subinterval
# Basis points cannot be at the knots
basis = []

for i in range(n):
    if i == 0:
        # Leftmost subinterval, first point can be at a
        basis.append(np.linspace(knots[i], knots[i + 1], m + 2)[:-1])
    elif i == n - 1:
        # Rightmost subinterval, last point can be at b
        basis.append(np.linspace(knots[i], knots[i + 1], m + 2)[1:])
    else:
        # Internal subintervals, cannot include the knots
        basis.append(np.linspace(knots[i], knots[i + 1], m + 2)[1:-1])

# Construct P_i matrices for each subinterval
# p^i_𝛼β = {(t_i𝛼-θ_{i-1})^β}, 𝛼=1,...,k_i, β=1,...,m
P_matrices = []  # 1,...,n
for i in range(n):
    prev_knot = knots[i]
    P_matrices.append(
        np.array(
            [[(t - prev_knot) ** beta for beta in range(1, m + 1)] for t in basis[i]]
        )
    )

# Construct Q_i matrices for each subinterval
# q^i_𝛼β = {(θ_i-θ_{i-1})^β}, β=1,...,m
Q_rows = []  # 1,...,n-1
for i in range(n - 1):
    prev_knot = knots[i]
    next_knot = knots[i + 1]
    Q_rows.append(
        np.array([[(next_knot - prev_knot) ** beta for beta in range(1, m + 1)]])
    )

# Construct full matrix (𝝲+2 rows) where 𝝲 is sum of basis points
gamma = [len(b) for b in basis]
gamma_sum = sum(gamma)

A = np.zeros((gamma_sum, gamma_sum))
b = np.zeros(gamma_sum)

row = 0
sign = 1

for interval in range(n):
    for r in range(gamma[interval]):
        # First element in each row is 1
        A[row, 0] = 1.0

        # Fill Q rows Q_1 to Q_{i-1} for the current interval
        for col_block in range(interval):
            len_block = 1 + col_block * m
            A[row, len_block : len_block + m] = Q_rows[col_block]

        # Fill this intervals P_i matrix
        len_block = 1 + interval * m
        A[row, len_block : len_block + m] = P_matrices[interval][r]

        # Fill delta column with alternating sign
        A[row, -1] = sign

        # Fill b with function values at basis points
        b[row] = f(basis[interval][r])

        sign *= -1
        row += 1

solution = np.linalg.solve(A, b)
a0 = solution[0]
a = solution[1:-1].reshape(n, m)
delta = solution[-1]


def idk(i, t):
    last_term = a0 if i == 0 else idk(i - 1, knots[i])
    return sum(a[i, j] * (t - knots[i]) ** (j + 1) for j in range(m)) + last_term


# Create spline S(A,t) from coefficients
for i in range(1, n):
    print(
        f"Continuity at knot {knots[i]}: S({idk(i-1, knots[i])}) = S({idk(i, knots[i])})"
    )
