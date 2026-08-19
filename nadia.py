import matplotlib.pyplot as plt
import numpy as np


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


# Construct P_i matrix for given subinterval i
# p^i_𝛼β = {(t_i𝛼-θ_{i-1})^β}, 𝛼=1,...,k_i, β=1,...,m
def build_P_matrix(i):
    prev_knot = knots[i]  # this is θ_{i-1}
    return np.array(
        [[(t - prev_knot) ** beta for beta in range(1, m + 1)] for t in basis[i]]
    )


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
    b = np.zeros(total_rows)

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
            b[row] = f(basis[interval][r])

            sign *= -1
            row += 1

    solution = np.linalg.solve(A, b)
    a0 = solution[0]
    a = solution[1:-1].reshape(n, m)
    delta = solution[-1]

    def S(i, t):
        last_term = a0 if i == 0 else S(i - 1, knots[i])
        return sum(a[i, j] * (t - knots[i]) ** (j + 1) for j in range(m)) + last_term

    return S, delta


def find_max_deviation_in_interval(f, S, i, knots, tol=1e-6, n_samples=10000):
    start, end = knots[i] + tol, knots[i + 1] - tol
    t_samples = np.linspace(start, end, n_samples)
    d_samples = np.abs(np.array([f(t) - S(i, t) for t in t_samples]))
    max_index = np.argmax(d_samples)
    return t_samples[max_index], f(t_samples[max_index]) - S(i, t_samples[max_index])


def find_max_deviation_overall(f, S, knots, n):
    interval_index = None
    t_star = None
    d_max = 0

    for i in range(n):
        t, d = find_max_deviation_in_interval(f, S, i, knots)
        if abs(d) > abs(d_max):
            d_max = d
            t_star = t
            interval_index = i
    return interval_index, t_star, d_max


def exchange(interval_index, t_star, d_max, f, S, basis):
    basis_points = basis[interval_index]
    t_star_sign = np.sign(d_max)

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
    return new_basis


def all_max_deviations_in_interval(f, S, i, knots, tol=1e-6, n_samples=10000):
    start, end = knots[i] + tol, knots[i + 1] - tol
    t_samples = np.linspace(start, end, n_samples)
    deviation = lambda t: f(t) - S(i, t)
    deviations = np.array([deviation(x) for x in t_samples])
    abs_deviations = np.abs(deviations)
    index = [0]
    for j in range(1, len(t_samples) - 1):
        if (
            abs_deviations[j] > abs_deviations[j - 1]
            and abs_deviations[j] > abs_deviations[j + 1]
        ):
            index.append(j)
    index.append(len(t_samples) - 1)
    return [(t_samples[j], deviations[j]) for j in index]


def find_alternance_points(f, S, knots, n):
    all_points = []
    for i in range(n):
        interval_points = all_max_deviations_in_interval(f, S, i, knots)
        all_points.extend(interval_points)

    global_max = max(abs(d) for _, d in all_points)
    filtered = np.array(
        [(t, d) for t, d in all_points if np.isclose(abs(d), global_max)]
    )

    # Handle duplicates
    unique = []
    for t, d in filtered:
        if not unique or t - unique[-1][0] > 1e-4:
            unique.append((t, d))

    pts = np.array([t for t, _ in unique])
    signs = np.array([np.sign(d) for _, d in unique])
    return pts, signs


# Necessary and suﬃcient optimality conditions for the spline S of degree m
def check_exit_1(f, S, knots, basis, m, tol=1e-6):
    pts, signs = find_alternance_points(f, S, knots, n)
    pts_and_signs = list(zip(pts, signs))

    def points_in_interval(i):
        start, end = knots[i] + tol, knots[i + 1] - tol
        return [(t, s) for t, s in pts_and_signs if start <= t <= end]

    def points_alternate(pts_and_signs):
        signs = [s for _, s in pts_and_signs]
        return len(signs) and all(
            signs[i] != signs[i + 1] for i in range(len(signs) - 1)
        )

    counts_per_interval = [len(points_in_interval(i)) for i in range(n)]

    # condition (i): in one subinterval, there is at least m+2 alternance points
    for i in range(n):
        pts_in_interval = points_in_interval(i)
        if len(pts_in_interval) >= m + 2 and points_alternate(pts_in_interval):
            return True, pts, signs

    for i in range(n):
        # Interval i fails (ii) point 1
        if counts_per_interval[i] < m + 1:
            continue
        for j in range(i, n):
            # Interval j fails (ii) point 2
            if counts_per_interval[j] < m + 1:
                continue
            # Point 3 (ii): at least m alternance points in the k−th interval i < k < j
            if not all(counts_per_interval[k] >= m for k in range(i + 1, j)):
                continue
            combined_pts = [
                (t, s)
                for t, s in pts_and_signs
                if knots[i] + tol <= t <= knots[j + 1] - tol
            ]
            # Passes condition (ii)
            if len(combined_pts) >= (m * (j - i - 1) + 2) and points_alternate(
                combined_pts
            ):
                return True, pts, signs

    return False, pts, signs


def plot(f, S, knots, basis, a, b, n, m, k, plot_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.arange(a, b, 0.01)
    ax.plot(t, f(t), color="slategray", label="f(t)")

    # approximation polynomial S(A,t) over each interval
    for i in range(n):
        t = np.arange(knots[i], knots[i + 1], 0.01)
        ax.plot(t, S(i, t), color="dodgerblue", label="S(A,t)" if i == 0 else None)
    # knots as vertical lines
    for i in knots:
        ax.axvline(i, color="red", ls=":", label="knots" if i == knots[0] else None)

    # basis points as dots
    for i in range(n):
        ax.plot(
            basis[i],
            f(basis[i]),
            "o",
            color="orange",
            label="basis points" if i == 0 else None,
        )

    ax.set_title(f"Degree-{m} approximation with {k} fixed knots")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("plot2.png")
    plt.show()


def f(t):
    return np.sin(t)


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
    optimal, pts, signs = check_exit_1(f, S, knots, n, m)
    print(
        f"Iteration 0: |delta|={abs(delta):.5f}  alternance_count={len(pts)}  optimal={optimal}"
    )

    plot(f, S, knots, basis, a, b, n, m, k, "nadia_plot_zero.png")

    max_iterations = 100

    for iteration in range(max_iterations):
        interval_index, t_star, max_dev = find_max_deviation_overall(f, S, knots, n)
        new_basis = exchange(interval_index, t_star, max_dev, f, S, basis)

        if new_basis is None:
            print(f"EXIT 2 (no valid exchange) at iteration {iteration}")
            break

        basis = new_basis
        S, delta = step_one(knots, basis, m, n, f)
        optimal, pts, signs = check_exit_1(f, S, knots, n, m)

        if optimal:
            print(
                f"EXIT 1 (spline is optimal) at iteration {iteration} with delta {abs(delta):.5f}"
            )
            print(f"alternance points: {np.round(pts, 6)}")
            break

    plot(f, S, knots, basis, a, b, n, m, k, "nadia_plot_final.png")
