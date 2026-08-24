import matplotlib.pyplot as plt
import numpy as np


# Form intial basis - m per internal subinterval, m+1 per endpoint subinterval
# Basis points cannot be at the knots except for endpoint intervals
def step_zero(knots, m, n):
    basis = []

    for i in range(n):
        start = knots[i] if i == 0 else knots[i] + 1e-3
        end = knots[i + 1] if i == n - 1 else knots[i + 1] - 1e-3
        counts = m + 1 if i == 0 or i == n - 1 else m
        basis.append(np.linspace(start, end, counts))
    return basis


# Construct P_i matrix for given subinterval i
# p^i_𝛼β = {(t_i𝛼-θ_{i-1})^β}, 𝛼=1,...,k_i, β=1,...,m
def build_P_matrix(i, basis, knots, m):
    prev_knot = knots[i]  # this is θ_{i-1}
    return np.array(
        [[(t - prev_knot) ** beta for beta in range(1, m + 1)] for t in basis[i]]
    )


# Construct Q_i matrix row for given subinterval i
# q^i_𝛼β = {(θ_i-θ_{i-1})^β}, β=1,...,m
def build_Q_row(i, knots, m):
    prev_knot = knots[i]
    next_knot = knots[i + 1]
    return np.array([[(next_knot - prev_knot) ** beta for beta in range(1, m + 1)]])


# step 1 : Construct full matrix (𝝲+2 rows) and solve for polynomial spline
def step_one(knots, basis, m, n, f):
    P_matrices = [build_P_matrix(i, basis, knots, m) for i in range(n)]
    Q_rows = [build_Q_row(i, knots, m) for i in range(n - 1)]

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


def deviation(f, S, i, t):
    return f(t) - S(i, t)


def find_max_deviation_in_interval(f, S, i, knots, tol=1e-6, n_samples=10000):
    """Find the point of maximum deviation in the interval [knots[i], knots[i+1]]"""
    start = knots[i] if i == 0 else knots[i] + tol
    end = knots[i + 1] if i == n - 1 else knots[i + 1] - tol
    t_samples = np.linspace(start, end, n_samples)
    d_samples = np.abs(np.array([deviation(f, S, i, t) for t in t_samples]))
    max_index = np.argmax(d_samples)
    return t_samples[max_index], f(t_samples[max_index]) - S(i, t_samples[max_index])


def find_max_deviation_overall(f, S, knots, n):
    """Find the interval and point of maximum deviation across all intervals"""
    interval = None
    t_star = None
    d_max = 0

    for i in range(n):
        t, d = find_max_deviation_in_interval(f, S, i, knots)
        if abs(d) > abs(d_max):
            d_max = d
            t_star = t
            interval = i
    return interval, t_star, d_max


def exchange(i, t_star, d_star, f, S, basis, knots, n, tol=1e-6):
    """Basis exchange rules

    For an interval i, assume there is a point t*, that is not an internal knot, where the absolute deviation is higher than the absolute deviation at any of the basis points in that interval. Then, if there is a point t~ in the basis of that interval, nearest to t*, such that the deviation at t~ has the same sign as the deviation at t*, then t* has to replace t~ in the basis of that interval. Otherwise, no exchange is possible.
    """
    # t_star cannot be an internal knot
    for j in range(1, n):
        if abs(t_star - knots[j]) < tol:
            return None

    basis_points = basis[i]

    # absolute deviation at t_star must be greater than the absolute deviation at any of the basis points in that interval
    basis_d_max = [abs(deviation(f, S, i, t)) for t in basis_points]
    if abs(d_star) <= max(basis_d_max) * (1 + tol):
        return None

    t_star_sign = np.sign(d_star)

    # Get basis points to the left and right of t_star
    left = basis_points[basis_points < t_star]
    right = basis_points[basis_points > t_star]
    left_pt = left[-1] if len(left) else None
    right_pt = right[0] if len(right) else None

    # Check if the deviation at the left or right basis point has the same sign as the deviation at t_star
    tilde_t = None
    if left_pt is not None and np.sign(deviation(f, S, i, left_pt)) == t_star_sign:
        tilde_t = left_pt
    elif right_pt is not None and np.sign(deviation(f, S, i, right_pt)) == t_star_sign:
        tilde_t = right_pt
    if tilde_t is None:
        return None

    # Replace tilde_t with t_star in basis points
    new_basis = [b.copy() for b in basis]
    new_interval_basis_pts = np.sort(
        np.append(basis_points[~np.isclose(basis_points, tilde_t)], t_star)
    )
    new_basis[i] = new_interval_basis_pts
    return new_basis


def find_local_peaks(f, S, i, knots, n, tol=1e-6, n_samples=10000):
    """Find local peaks of the absolute deviation in the interval [knots[i], knots[i+1]]. find_alternance_points uses this to find true max"""

    # Add a small tolerance to avoid sampling exactly at the knots for internal intervals
    start = knots[i] if i == 0 else knots[i] + tol
    end = knots[i + 1] if i == n - 1 else knots[i + 1] - tol

    # Take n_samples evenly spaced points in the interval
    t_samples = np.linspace(start, end, n_samples)
    d_samples = np.array([deviation(f, S, i, t) for t in t_samples])
    abs_d_samples = np.abs(d_samples)

    # Find local maxima in the absolute deviation samples
    index = []
    if abs_d_samples[0] > abs_d_samples[1]:
        index.append(0)
    for j in range(1, len(t_samples) - 1):
        if (
            abs_d_samples[j] > abs_d_samples[j - 1]
            and abs_d_samples[j] > abs_d_samples[j + 1]
        ):
            index.append(j)

    if abs_d_samples[-1] > abs_d_samples[-2]:
        index.append(len(t_samples) - 1)

    return [(t_samples[j], d_samples[j]) for j in index]


def find_alternance_points(f, S, knots, n):
    """Find all alternance points (points where absolute deviation is maximal) in the spline approximation"""
    all_points = []
    for i in range(n):
        interval_points = find_local_peaks(f, S, i, knots, n)
        all_points.extend(interval_points)

    global_max = max(abs(d) for _, d in all_points)
    filtered = [
        (t, d) for t, d in all_points if np.isclose(abs(d), global_max, rtol=1e-2)
    ]

    # Handle duplicates
    unique = []
    for t, d in filtered:
        if not unique or t - unique[-1][0] > 1e-4:
            unique.append((t, d))

    pts = np.array([t for t, _ in unique])
    signs = np.array([np.sign(d) for _, d in unique])
    return pts, signs, global_max


def check_exit_1(f, S, knots, n, m, tol=1e-6):
    """Check the necessary and sufficient optimality conditions for the spline S of degree m (Theorem 1.10, Tarashnin, EXIT 1 in the algorithm)."""
    pts, signs, global_max = find_alternance_points(f, S, knots, n)
    pts_and_signs = list(zip(pts, signs))

    def points_in_interval(i):
        """Return all alternance points in the i-th interval (knots[i], knots[i+1])"""
        start = knots[i] if i == 0 else knots[i] + tol
        end = knots[i + 1] if i == n - 1 else knots[i + 1] - tol
        return [(t, s) for t, s in pts_and_signs if start <= t <= end]

    def points_alternate(pts_and_signs):
        """Return True if the points alternate in sign, false otherwise"""
        signs = [s for _, s in pts_and_signs]
        return len(signs) >= 2 and all(
            signs[i] != signs[i + 1] for i in range(len(signs) - 1)
        )

    # Number of alternance points in each interval
    counts_per_interval = [len(points_in_interval(i)) for i in range(n)]

    # condition (i): in one subinterval, there is at least m+2 alternance points
    for i in range(n):
        pts_in_interval = points_in_interval(i)
        if len(pts_in_interval) >= m + 2 and points_alternate(pts_in_interval):
            return True, pts, signs, (i, i)

    for i in range(n):
        # Interval i fails (ii) point 1: less than m+1 alternance points in interval i
        if counts_per_interval[i] < m + 1:
            continue
        for j in range(i, n):
            # Interval j fails (ii) point 2: less than m+1 alternance points in interval j
            if counts_per_interval[j] < m + 1:
                continue
            # Interval k fails (ii) point 3 - less than m alternance points in the k−th interval i < k < j
            if not all(counts_per_interval[k] >= m for k in range(i + 1, j)):
                continue
            combined_pts = [
                (t, s)
                for t, s in pts_and_signs
                if knots[i] + tol <= t <= knots[j + 1] - tol
            ]
            # Passes condition (ii)
            if len(combined_pts) >= (m * (j - i + 1) + 2) and points_alternate(
                combined_pts
            ):
                return (True, pts, signs, (i, j))

    return False, pts, signs, None


def plot(f, S, knots, basis, a, b, n, m, k, plot_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.arange(a, b, 0.01)
    ax.plot(t, f(t), color="slategray", label="f(t)")

    # approximation polynomial S(A,t) over each interval
    for i in range(n):
        t = np.arange(knots[i], knots[i + 1], 0.01)
        ax.plot(t, S(i, t), color="dodgerblue", label="S(A,t)" if i == 0 else None)

    # basis points as vertical dashed lines
    for i in range(n):
        for j in basis[i]:
            ax.axvline(
                j,
                color="grey",
                ls="--",
                lw=1,
                label="basis points" if i == 0 and j == basis[0][0] else None,
            )
    # knots as vertical lines
    for i in knots:
        ax.axvline(i, color="red", ls=":", label="knots" if i == knots[0] else None)

    ax.set_title(f"Degree-{m} approximation with {k} fixed knots")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_name)
    plt.show()


if __name__ == "__main__":

    def f(t):
        return np.sin(t)

    a, b = 0, 6
    k = 2  # number of free knots (not including a and b)
    m = 2  # degree of polynomial to fit in each subinterval (assumed constant throughout)
    n = k + 1  # number of subintervals
    tol = 1e-6

    # Choose intial knots as equidistant points
    knots = np.linspace(a, b, k + 2)  # θ_0,...,θ_n and internal θ_1,...,θ_{n-1}

    # Choose initial basis
    basis = step_zero(knots, m, n)

    # Construct polynomial spline S
    S, delta = step_one(knots, basis, m, n, f)
    optimal, pts, signs, chain = check_exit_1(f, S, knots, n, m)

    max_iterations = 100

    for _ in range(max_iterations):
        interval_index, t_star, max_dev = find_max_deviation_overall(f, S, knots, n)

        new_basis = exchange(interval_index, t_star, max_dev, f, S, basis, knots, n)

        if new_basis is None:
            print("EXIT 2 (no valid exchange).")
            break

        basis = new_basis
        S, delta = step_one(knots, basis, m, n, f)
        optimal, pts, signs, chain = check_exit_1(f, S, knots, n, m)

        if optimal:
            print("EXIT 1 (spline is optimal).")
            break

    print(f"Delta: {abs(delta):.5f}")
    print(f"Max deviation: {abs(max_dev):.5f}")
    print(f"Basis points: {np.round(np.concatenate(basis), 6)}")
    if chain is not None:
        ci, cj = chain
        print(f"minimal chain: interval {ci}-{cj}  (length {cj-ci+1})")

    plot(f, S, knots, basis, a, b, n, m, k, "nadia_plot.png")
