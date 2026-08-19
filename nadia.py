import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.sin(x)


# Form intial basis - m per internal subinterval, m+1 per endpoint subinterval
# Basis points cannot be at the knots except for endpoint intervals
def step_zero(knots, m, n):
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
    return basis


# for i in range(n):
#     prev_knot = knots[i]  # this is θ_{i-1}
#     P_matrices.append(
#         # β=1 to m
#         np.array(
#             [[(t - prev_knot) ** beta for beta in range(1, m + 1)] for t in basis[i]]
#         )
#     )


# Construct P_i matrices for given subinterval i
# p^i_𝛼β = {(t_i𝛼-θ_{i-1})^β}, 𝛼=1,...,k_i, β=1,...,m
# P_matrices = []  # 0,...,n-1 (1,...,n)
def build_P_matrix(i):
    prev_knot = knots[i]  # this is θ_{i-1}
    return np.array(
        [[(t - prev_knot) ** beta for beta in range(1, m + 1)] for t in basis[i]]
    )


# Q_rows = []  # 1,...,n-1
# for i in range(n - 1):
#     prev_knot = knots[i]
#     next_knot = knots[i + 1]
#     Q_rows.append(
#         np.array([[(next_knot - prev_knot) ** beta for beta in range(1, m + 1)]])
#     )


# Construct Q_i matrix row for given subinterval i
# q^i_𝛼β = {(θ_i-θ_{i-1})^β}, β=1,...,m
def build_Q_row(i):
    prev_knot = knots[i]
    next_knot = knots[i + 1]
    return np.array([[(next_knot - prev_knot) ** beta for beta in range(1, m + 1)]])


# step 1 : Construct full matrix (𝝲+2 rows) and solve for polynomial spline
def step_one(knots, basis, m, n, f):
    P_matrices = [build_P_matrix(i) for i in range(n)]
    Q_rows = [build_Q_row(i) for i in range(n - 1)]

    basis_counts = [len(b) for b in basis]
    total_rows = sum(basis_counts)

    A = np.zeros((total_rows, total_rows))
    rhs = np.zeros(total_rows)

    row = 0
    sign = 1

    for interval in range(n):
        for r in range(basis_counts[interval]):
            # First element in each row is 1
            A[row, 0] = 1.0

            # Fill Q rows Q_1 to Q_{i-1} for the current interval
            for col_start in range(interval):
                len_block = 1 + col_start * m
                A[row, len_block : len_block + m] = Q_rows[col_start]

            # Fill this intervals P_i matrix
            len_block = 1 + interval * m
            A[row, len_block : len_block + m] = P_matrices[interval][r]

            # Fill delta column with alternating sign
            A[row, -1] = sign

            # Fill b with function values at basis points
            rhs[row] = f(basis[interval][r])

            sign *= -1
            row += 1

    solution = np.linalg.solve(A, rhs)
    coeffs_0 = solution[0]
    coeffs = solution[1:-1].reshape(n, m)
    delta = solution[-1]

    def S(i, t):
        last_term = coeffs_0 if i == 0 else S(i - 1, knots[i])
        return (
            sum(coeffs[i, j] * (t - knots[i]) ** (j + 1) for j in range(m)) + last_term
        )

    return S, delta


def find_max_deviation_in_interval(f, S, i, knots, tol=1e-6, n_samples=10000):
    start, end = knots[i] + tol, knots[i + 1] - tol
    x_samples = np.linspace(start, end, n_samples)
    e_samples = np.abs(np.array([f(x) - S(i, x) for x in x_samples]))
    max_index = np.argmax(e_samples)
    return x_samples[max_index], f(x_samples[max_index]) - S(i, x_samples[max_index])


def find_max_deviation_overall(f, S, knots, n):
    interval_index = None
    t_star = None
    max_dev = 0

    for i in range(n):
        t, dev = find_max_deviation_in_interval(f, S, i, knots)
        if abs(dev) > abs(max_dev):
            max_dev = dev
            t_star = t
            interval_index = i
    return interval_index, t_star, max_dev


def exchange(interval_index, t_star, max_dev, f, S, basis):
    basis_points = basis[interval_index]
    t_star_sign = np.sign(max_dev)

    deviation = lambda t: f(t) - S(interval_index, t)

    left = basis_points[basis_points < t_star]
    right = basis_points[basis_points > t_star]
    left_pt = left[-1] if len(left) else None
    right_pt = right[0] if len(right) else None

    tilde_t_val = None
    if left_pt is not None and np.sign(deviation(left_pt)) == t_star_sign:
        tilde_t_val = left_pt
    elif right_pt is not None and np.sign(deviation(right_pt)) == t_star_sign:
        tilde_t_val = right_pt
    if tilde_t_val is None:
        return None

    # Replace tilde_t with t_star in basis points
    new_basis = [b.copy() for b in basis]
    new_interval_basis_pts = np.sort(
        np.append(basis_points[~np.isclose(basis_points, tilde_t_val)], t_star)
    )
    new_basis[interval_index] = new_interval_basis_pts
    return new_basis, tilde_t_val


def all_max_deviations_in_interval(f, S, i, knots, tol=1e-6, n_samples=10000):
    start, end = knots[i] + tol, knots[i + 1] - tol
    x_samples = np.linspace(start, end, n_samples)
    deviation = lambda t: f(t) - S(i, t)
    deviations = np.array([deviation(x) for x in x_samples])
    abs_deviations = np.abs(deviations)
    index = [0]
    for j in range(1, len(x_samples) - 1):
        if (
            abs_deviations[j] > abs_deviations[j - 1]
            and abs_deviations[j] > abs_deviations[j + 1]
        ):
            index.append(j)
    index.append(len(x_samples) - 1)
    return [(x_samples[j], deviations[j]) for j in index]


def find_alternance_points(f, S, knots, n):
    all_points = []
    for i in range(n):
        interval_points = all_max_deviations_in_interval(f, S, i, knots)
        all_points.extend(interval_points)

    global_max = max(abs(dev) for _, dev in all_points)
    filtered = [(t, dev) for t, dev in all_points if np.isclose(abs(dev), global_max)]

    # Handle duplicates
    deduped = []
    for t, v in filtered:
        if not deduped or t - deduped[-1][0] > 1e-4:
            deduped.append((t, v))
    pts = np.array([t for t, _ in deduped])
    signs = np.array([np.sign(v) for _, v in deduped])
    return pts, signs, global_max


# Necessary and suﬃcient optimality conditions for the spline S of degree m
def check_exit_1(f, S, knots, basis, m, knot_tol=1e-6):
    # In one subinterval, there is at least m+2 alternance points
    pts, signs, global_max = find_alternance_points(f, S, knots, n)
    pts_and_signs = list(zip(pts, signs))

    def points_in_interval(i):
        start, end = knots[i] + knot_tol, knots[i + 1] - knot_tol
        return [(t, s) for t, s in pts_and_signs if start <= t <= end]

    def points_alternate(pts_and_signs):
        signs = [s for _, s in pts_and_signs]
        return len(signs) and all(
            signs[i] != signs[i + 1] for i in range(len(signs) - 1)
        )

    counts_per_interval = [len(points_in_interval(i)) for i in range(n)]

    for i in range(n):
        pts_in_interval = points_in_interval(i)
        if len(pts_in_interval) >= m + 2 and points_alternate(pts_in_interval):
            return True, pts, signs, global_max

    for i in range(n):
        if counts_per_interval[i] < m + 1:
            continue
        for j in range(i, n):
            if counts_per_interval[j] < m + 1:
                continue
            if not all(counts_per_interval[k] >= m for k in range(i + 1, j)):
                continue
            combined_pts = [
                (t, s)
                for t, s in pts_and_signs
                if knots[i] + knot_tol <= t <= knots[j + 1] - knot_tol
            ]
            needed = m * (j - i - 1) + 2
            if len(combined_pts) >= needed and points_alternate(combined_pts):
                return True, pts, signs, global_max

    return False, pts, signs, global_max


# # print basis points
# for i in range(n):
#     print(f"Basis points for interval {i}: {basis[i]}")

# print(f"Delta: {delta}")


# # check spline deviates by same abs value from f and signs alternate (step 1 Nadia Poussin paper)
# expected_sign = 1
# for i in range(n):
#     for t in basis[i]:
#         deviation = f(t) - P(i, t)
#         expected = expected_sign * delta
#         assert np.isclose(deviation, expected)
#         expected_sign *= -1

if __name__ == "__main__":
    a, b = 0, 6
    k = 2  # number of free knots (not including a and b)
    m = 2  # degree of polynomial to fit in each subinterval (assumed constant throughout)
    n = k + 1  # number of subintervals

    # Choose intial knots as equidistant points
    knots = np.linspace(a, b, k + 2)  # θ_0,...,θ_n and internal θ_1,...,θ_{n-1}

    # Choose initial basis
    basis = step_zero(knots, m, n)

    S, delta = step_one(knots, basis, m, n, f)
    optimal, pts, signs, gmax = check_exit_1(f, S, knots, n, m)
    print(
        f"round 0: |delta|={abs(delta):.6f}  alternance_count={len(pts)}  optimal={optimal}"
    )

    max_iterations = 100

    for iteration in range(max_iterations):
        interval_index, t_star, max_dev = find_max_deviation_overall(f, S, knots, n)
        result = exchange(interval_index, t_star, max_dev, f, S, basis)

        if result is None:
            print(f"EXIT 2 (no valid exchange) at iteration {iteration}")
            break

        basis, tilde_t_val = result
        S, delta = step_one(knots, basis, m, n, f)
        optimal, pts, signs, gmax = check_exit_1(f, S, knots, n, m)
        print(
            f"iteration {iteration}: |delta|={abs(delta):.6f}  alternance_count={len(pts)}  optimal={optimal}"
        )
        if optimal:
            print(f"  -> EXIT 1 at iteration {iteration}, spline is optimal")
            print(f"  alternance points: {np.round(pts, 5)}")
            break

# fig, ax = plt.subplots(figsize=(10, 6))

# # original function f(x)
# x = np.arange(a, b, 0.01)
# ax.plot(x, f(x), color="slategray", label="f(x)")

# for i in range(n):
#     # approximation polynomial P(x) over each interval
#     x = np.arange(knots[i], knots[i + 1], 0.01)
#     ax.plot(x, P(i, x), color="dodgerblue", label="P(x)" if i == 0 else None)

# # knots as vertical lines
# for x in knots:
#     ax.axvline(x, color="red", ls=":", label="knots" if x == knots[0] else None)

# # basis points as dots
# for i in range(n):
#     ax.plot(
#         basis[i],
#         f(basis[i]),
#         "o",
#         color="orange",
#         label="basis points" if i == 0 else None,
#     )


# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_title(f"Degree-{m} approximation with {k} fixed knots")
# ax.legend(loc="best")
# fig.tight_layout()
# fig.savefig("plot2.png")
# plt.show()
